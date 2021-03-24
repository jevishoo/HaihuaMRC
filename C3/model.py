import torch
import torch.nn as nn
from nezha.nezha_modeling import BertPreTrainedModel, BertModel, BertLayerNorm
from down_layer.bilstm import BiLSTM


class BertForMRC(BertPreTrainedModel):

    def __init__(self, config, num_choices):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_choices = num_choices
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dense_final = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(True))
        # 动态权重
        self.dym = True
        self.dym_weight = nn.Parameter(torch.ones((config.num_hidden_layers, 1, 1, 1)), requires_grad=True)
        self.pool_weight = nn.Parameter(torch.ones((2, 1, 1, 1)), requires_grad=True)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BertLayerNorm):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                module.bias.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

        self.bilstm = BiLSTM(embedding_size=config.hidden_size, hidden_size=256, num_layers=2,
                             num_classes=config.hidden_size)

        self.cls = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_dym_layers(self, outputs):
        layer_logits = []
        all_encoder_layers = outputs[1:]
        for i, layer in enumerate(all_encoder_layers):
            layer_logits.append(self.classifier(layer))

        layer_logits = torch.cat(layer_logits, 2)

        layer_dist = nn.functional.softmax(layer_logits)
        seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out)
        pooled_output = torch.squeeze(pooled_output, 2)

        word_embed = self.dense_final(pooled_output)
        dym_layer = word_embed
        return dym_layer

    def get_weight_layers(self, outputs):
        hidden_stack = torch.stack(outputs[1:], dim=0)
        sequence_output = torch.sum(hidden_stack + self.dym_weight, dim=0)
        return sequence_output

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, return_logits=False):
        # input_ids: [bs,num_choice,seq_l]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))  # flat_input_ids: [bs*num_choice,seq_l]
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        encoded_layers, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask,
                                                  output_all_encoded_layers=True)

        if self.dym:
            sequence_output = self.get_dym_layers(encoded_layers)
        else:
            sequence_output = self.get_weight_layers(encoded_layers)

        lstm_output = self.bilstm(sequence_output)
        pooled_output = self.dropout(lstm_output)
        logits = self.cls(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.view(-1, self.num_choices)  # logits: (bs, num_choice)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            labels = labels.squeeze(1)  # 1*bs
            loss = loss_fct(reshaped_logits, labels)
            if return_logits:
                return loss, reshaped_logits
            else:
                return loss
        else:
            return reshaped_logits
