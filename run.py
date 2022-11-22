import os
import logging
import numpy as np
import torch
from tools.progressbar import ProgressBar
from tools.collate_fn import collate_fn
from tools.warmup import WarmupLinearSchedule
from tools.multi_compute_metrics import compute_metrics
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
import argparse
import processor
from model import BertForMyMultiTask
from transformers.models.bert import BertTokenizer, BertConfig
from torch.optim.adamw import AdamW

import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger()


def load_and_cache_examples(data_type, tokenizer, max_label_list):
    if data_type == 'train':
        examples = processor.loader.get_examples(args.data_dir, "train")
    elif data_type == 'dev':
        examples = processor.loader.get_examples(args.data_dir, "dev")
    else:
        examples = processor.loader.get_examples(args.data_dir, "test")

    features = processor.loader.convert_examples_to_features(examples, tokenizer=tokenizer,
                                                             max_label_list=max_label_list,
                                                             max_length=args.max_length)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_labels = torch.tensor([list(f.label) for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset


def train(args, train_dataset, model, tokenizer):
    args.train_batch_size = args.train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = args.max_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  train batch size  = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    print(int(args.num_train_epochs))
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'token_type_ids': batch[2],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print(" ")
                    # Log metrics
                    evaluate(args, model, tokenizer, max_label_list)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(save_directory=output_dir)
        print(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, max_label_list, prefix=""):
    eval_outputs_dir = args.output_dir
    eval_batch_size = args.eval_batch_size
    results = {}
    eval_dataset = load_and_cache_examples(data_type="dev", tokenizer=tokenizer, max_label_list=max_label_list)
    if not os.path.exists(eval_outputs_dir):
        os.makedirs(eval_outputs_dir)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, eval_batch_size, sampler=eval_sampler, collate_fn=collate_fn)

    # 开始预测
    logger.info("********* Running evaluation {} ********".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_all = []
    out_label_ids = []
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'token_type_ids': batch[2],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            outputs = model(**inputs)
            temp_eval_loss, logits = outputs[0], outputs[1:]
            eval_loss += temp_eval_loss.mean().item()
        nb_eval_steps += 1
        preds_all.append([logit.detach().cpu().numpy() for logit in logits])
        out_label_ids.append(
            [inputs['labels'][:, int(i)].detach().cpu().numpy() for i in range(len(inputs['labels'][0]))])
        pbar(step)

    if "cuda" in str(args.device):
        torch.cuda.empty_cache()
    eval_loss = eval_loss / nb_eval_steps

    preds_all = [np.argmax(preds, axis=2) for preds in preds_all]

    pres_purpose = [i for j in [_[0].tolist() for _ in preds_all] for i in j]
    pres_type = [i for j in [_[1].tolist() for _ in preds_all] for i in j]
    pres_15you = [i for j in [_[2].tolist() for _ in preds_all] for i in j]
    out_label_id_purpose = [i for j in [_[0].tolist() for _ in out_label_ids] for i in j]
    out_label_id_type = [i for j in [_[1].tolist() for _ in out_label_ids] for i in j]
    out_label_id_15you = [i for j in [_[2].tolist() for _ in out_label_ids] for i in j]

    result = [compute_metrics(pres_purpose, out_label_id_purpose),
              compute_metrics(pres_type, out_label_id_type),
              compute_metrics(pres_15you, out_label_id_15you)]

    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("******** Eval results {} ********".format(prefix))

    for i, result_ in enumerate(result):
        for key in sorted(result_.keys()):
            # 无需打印“acc_dict”
            if key == "acc_dict":
                continue
            logger.info(" dev: %s = %s", key, str(result_[key]))
            print(str(result_[key]))
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--do_train", action="store_true", help="wether to run train")
    parser.add_argument("--data_dir", default="./dataset", type=str)
    parser.add_argument("--tokenizer", default="bert-base-chinese", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str)
    parser.add_argument("--classifier_1_nums", default=3, type=int)
    parser.add_argument("--classifier_2_nums", default=3, type=int)
    parser.add_argument("--classifier_3_nums", default=3, type=int)
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int)
    parser.add_argument('--logging_steps', type=int, default=16, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--output_dir", default="./outputs/", type=str)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=2, type=int,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    args = parser.parse_args()

    max_label_list = [str(i) for i in
                      range(int(max(args.classifier_1_nums, args.classifier_2_nums, args.classifier_3_nums)))]
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    config = BertConfig.from_pretrained(args.model_name_or_path)
    model = BertForMyMultiTask.from_pretrained(args.model_name_or_path, config=config, args=args)
    model.unfreeze(1, 6)
    model.to(args.device)

    # if args.do_train:
    if True:
        print("start...")
        train_dataset = load_and_cache_examples(data_type='train', tokenizer=tokenizer, max_label_list=max_label_list)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
