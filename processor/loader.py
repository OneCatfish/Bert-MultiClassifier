import copy
import json
import os.path
import pandas as pd


def read_json(input_file):
    with open(input_file, "r", encoding="utf8") as f:
        reader = f.readlines()
        lines = [json.loads(_.strip()) for _ in reader]
    return lines


class InputExample:
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):

    def __init__(self, input_ids, attention_mask, token_type_ids, label, input_len):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def create_examples(lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
        text = line["sentence"]
        label = str(line["label"] if set_type != "test" else "0")
        examples.append(InputExample(text=text, label=label))

    return examples


def get_examples(input_file, set_type):
    lines = read_json(os.path.join(input_file, set_type+".json"))
    return create_examples(lines, set_type=set_type)


def convert_examples_to_features(example, tokenizer, max_length, max_label_list,
                                 pad_token=0, mask_padding=0, pad_token_segment_id=0):
    # 多分类标签输出
    label_map = {label: i for i, label in enumerate(max_label_list)}

    features = []
    for ex_index, example in enumerate(example):
        inputs = tokenizer(text=example.text,
                           add_special_tokens=True,
                           max_length=max_length,
                           truncation=True)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        input_length = len(input_ids)
        padding_length = max_length - input_length

        input_ids = input_ids + [pad_token] * padding_length
        attention_mask = attention_mask + [mask_padding] * padding_length
        token_type_ids = token_type_ids + [pad_token_segment_id] * padding_length

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)
        label = (label_map[i] for i in example.label.split(","))

        features.append(InputFeatures(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      label=label,
                                      input_len=input_length))

    return features
