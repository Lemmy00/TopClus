import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from transformers import AutoTokenizer, AutoModel
from transformers import BertPreTrainedModel, BertModel

class AutoEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims):
        super(AutoEncoder, self).__init__()
        self.encoder_layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.encoder_layers.append(layer)
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.decoder_layers = []
        hidden_dims.reverse()
        dims = hidden_dims + [input_dim]
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.decoder_layers.append(layer)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, dim=-1)
        x_bar = self.decoder(z)
        return x_bar, z

    def decode(self, z):
        z = F.normalize(z, dim=-1)
        return self.decoder(z)


class TopClusModel(nn.Module):

    def __init__(self, config, input_dim, hidden_dims, n_clusters, kappa):
        super(TopClusModel, self).__init__()
        self.n_clusters = n_clusters
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.topic_emb = Parameter(torch.Tensor(n_clusters, hidden_dims[-1]))
        self.bert = AutoModel.from_pretrained(config, add_pooling_layer=False)
        self.ae = AutoEncoder(input_dim, hidden_dims)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.kappa = kappa
        self.v = Parameter(torch.rand(config.hidden_size))
        torch.nn.init.xavier_normal_(self.topic_emb.data)
        torch.nn.init.xavier_normal_(self.v.data)
        torch.nn.init.xavier_normal_(self.dense.weight)
        for param in self.bert.parameters():
            param.requires_grad = False

    def get_max_topic(self):
        return self.n_clusters

    def cluster_assign(self, z):
        self.topic_emb.data = F.normalize(self.topic_emb.data, dim=-1)
        sim = torch.matmul(z, self.topic_emb.t()) * self.kappa
        p = F.softmax(sim, dim=-1)
        return p

    def topic_sim(self, z):
        self.topic_emb.data = F.normalize(self.topic_emb.data, dim=-1)
        sim = torch.matmul(z, self.topic_emb.t())
        return sim

    # return initialized latent word embeddings
    def init_emb(self, input_ids, attention_mask, valid_pos):
        self.bert.eval()
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask)
        last_hidden_states = bert_outputs[0]
        attention_mask[:, 0] = 0
        attn_mask = valid_pos != 0
        input_embs = last_hidden_states[attn_mask]
        _, z = self.ae(input_embs)
        return z

    def forward(self, input_ids, attention_mask, valid_pos=None, pretrain=False):
        self.bert.eval()
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask)
        last_hidden_states = bert_outputs[0]

        if pretrain:
            attn_mask = attention_mask != 0
            input_embs = last_hidden_states[attn_mask]
            output_embs, _ = self.ae(input_embs)
            return input_embs, output_embs
        else:
            assert valid_pos is not None, "valid_pos should not be None in clustering mode!"
        attention_mask[:, 0] = 0
        sum_emb = (last_hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
        valid_num = attention_mask.sum(dim=-1, keepdim=True)
        avg_doc_emb = sum_emb / valid_num
        trans_states = self.dense(last_hidden_states)
        trans_states = self.activation(trans_states)
        attn_logits = torch.matmul(trans_states, self.v)
        attention_mask[:, 0] = 0
        attn_mask = attention_mask == 0
        attn_logits.masked_fill_(attn_mask, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)
        doc_emb = (last_hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        attn_mask = valid_pos != 0
        input_embs = last_hidden_states[attn_mask]
        output_embs, z_word = self.ae(input_embs)
        _, z_doc = self.ae(doc_emb)
        p_doc = self.cluster_assign(z_doc)
        p_word = self.cluster_assign(z_word)
        dec_topic = self.ae.decode(self.topic_emb)
        rec_doc_emb = torch.matmul(p_doc, dec_topic)
        return avg_doc_emb, input_embs, output_embs, rec_doc_emb, p_word

    def inference(self, input_ids, attention_mask):
        self.eval()
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask)
        last_hidden_states = bert_outputs[0]
        attention_mask[:, 0] = 0
        trans_states = self.dense(last_hidden_states)
        trans_states = self.activation(trans_states)
        attn_logits = torch.matmul(trans_states, self.v)
        attention_mask[:, 0] = 0
        attn_mask = attention_mask == 0
        attn_logits.masked_fill_(attn_mask, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)
        doc_emb = (last_hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        valid_word_embs = last_hidden_states[~attn_mask]
        valid_word_ids = input_ids[~attn_mask]
        _, z = self.ae(valid_word_embs)
        sim = self.topic_sim(z)
        _, z = self.ae(doc_emb)
        return z, valid_word_ids, sim
    
    def get_latent_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            self.eval()
            bert_outputs = self.bert(input_ids,
                                    attention_mask=attention_mask)
            last_hidden_states = bert_outputs[0]
            attention_mask[:, 0] = 0
            trans_states = self.dense(last_hidden_states)
            trans_states = self.activation(trans_states)
            attn_logits = torch.matmul(trans_states, self.v)
            attention_mask[:, 0] = 0
            attn_mask = attention_mask == 0
            attn_logits.masked_fill_(attn_mask, float('-inf'))
            attn_weights = F.softmax(attn_logits, dim=-1)
            doc_emb = (last_hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)

            _, z_doc = self.ae(doc_emb)
            return z_doc
    
    def encode(self, data, device='cpu', batch_size=32):
        pass
    
    def soft_clustering(self, latent_embeddings, top_p=0.90):
        with torch.no_grad():
            self.eval()
            p_doc = self.cluster_assign(latent_embeddings)

            # Soft clustering of comments with respect probabilities of topics.
            # Each comment is assigned to the topics with the top_p probabilities.
            # In other words, the comment is assigned to the topics with the highest 
            # probabilities until the cumulative probability exceeds top_p.
            sorted_p, indices = torch.sort(p_doc, dim=-1, descending=True)
            margin_indices = torch.sum(torch.cumsum(sorted_p, dim=-1) < top_p, dim=-1) + 1
            soft_clusters = torch.zeros_like(p_doc)
            for i in range(p_doc.size(0)):
                soft_clusters[i, indices[i, :margin_indices[i]]] = 1

            return soft_clusters
