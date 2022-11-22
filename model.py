import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel


class BertForMyMultiTask(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertForMyMultiTask, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier_1 = nn.Linear(config.hidden_size, args.classifier_1_nums)
        self.classifier_2 = nn.Linear(config.hidden_size, args.classifier_2_nums)
        self.classifier_3 = nn.Linear(config.hidden_size, args.classifier_3_nums)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        # 取Bert模型的输出作为分类器的输入
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits_1 = self.classifier_1(pooled_output)
        logits_2 = self.classifier_2(pooled_output)
        logits_3 = self.classifier_3(pooled_output)

        # 计算各个类别的loss
        loss_fct = CrossEntropyLoss()
        loss_1 = loss_fct(logits_1, labels[:, 0])
        loss_2 = loss_fct(logits_2, labels[:, 1])
        loss_3 = loss_fct(logits_3, labels[:, 2])

        multi_loss = loss_1 + loss_2 + loss_3

        outputs = (multi_loss,) + (logits_1, logits_2, logits_3)
        return outputs

    def unfreeze(self, start_layer, end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())

        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b

        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)

        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer + 1):
            set_trainable(self.bert.encoder.layer[i], True)
