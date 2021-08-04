import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import logging
import json
import numpy as np

def get_run_name(args):
    return [f'{param}={val}' for param, val in vars(args).items()]

def load_vocab(path, max_size, pad_token, unk_token):
    vocab = {pad_token: 0 , unk_token: 1}
    with open(path, 'r', encoding='utf-8') as f:
        for i, tok in enumerate(f):
            tok = tok.split('\t')[0]
            if i >= max_size:
                return vocab
            if tok not in vocab:
                vocab[tok.strip()] = len(vocab)
    return vocab


def numericalize(vocab, unk_token, max_length, tokens):
    idxs = [vocab.get(t, vocab[unk_token]) for t in tokens[:max_length]]
    return idxs

def count_parameters(models):
    if isinstance(models, list):
        return sum([count_parameters(model) for model in models])
    else:
        return sum(p.numel() for p in models.parameters() if p.requires_grad)

def load_lm_data(path, tokenizer, data):
    if data == 'desc':
        data = 'docstring'
    all_tokens = []
    with open(path, 'r') as f:
        for line in f:
            tokens = json.loads(line)
            tokens = tokens[f'{data}_tokens']
            tokens = tokenizer(tokens)
            all_tokens += tokens
    return torch.LongTensor(all_tokens)

def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts.
    n_batches = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, n_batches * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data

class SoftmaxLossRet(nn.Module):
    def __init__(self,
                 device):
        super().__init__()

        self.device = device

    def forward(self, enc_code, enc_desc):

        #enc_code = [batch size, enc dim]
        #enc_desc = [batch size, enc dim]

        enc_desc = enc_desc.permute(1, 0)

        #enc_desc = [enc dim, batch size]

        similarity = torch.matmul(enc_code, enc_desc)

        #similarity = [batch size, batch size]

        classes = torch.arange(similarity.shape[0]).to(self.device)

        loss = F.cross_entropy(similarity, classes)

        with torch.no_grad():
            mrr = mrr_metric_ret(similarity)

        return loss, mrr

def mrr_metric_ret(similarity):
    correct_scores = torch.diagonal(similarity)
    compared_scores = similarity >= correct_scores.unsqueeze(-1)
    rr = 1 / compared_scores.float().sum(-1)
    mrr = rr.mean()
    return mrr

class SoftmaxLossPred(nn.Module):
    def __init__(self,
                 device):
        super().__init__()

        self.device = device

    def forward(self, enc_code, labels):

        #enc_code = [batch size, out dim]
        #labels = [batch size]

        labels = labels.squeeze(0)

        loss = F.cross_entropy(enc_code, labels)

        with torch.no_grad():
            mrr = mrr_metric_pred(enc_code, labels)

        return loss, mrr

def mrr_metric_pred(enc_code, labels):
    n_classes = enc_code.shape[-1]
    one_hot = F.one_hot(labels, n_classes)
    actual_score = (enc_code * one_hot).sum(-1).unsqueeze(-1).repeat(1, n_classes)
    compared_scores = enc_code >= actual_score
    rr = 1/compared_scores.float().sum(-1)
    mrr = rr.mean()
    return mrr

def make_mask(sequence, pad_idx):
    mask = (sequence != pad_idx).permute(1, 0)
    return mask

def truncated_normal_(tensor, mean=0, std=1):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

def initialize_transformer(m):
    if isinstance(m, nn.LayerNorm):
        pass
    elif hasattr(m, 'weight'):
        truncated_normal_(m.weight.data, std=0.02)


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def default_init(tensor):
    if tensor.ndimension() == 1:
        nn.init.constant_(tensor, val=0.0)
    else:
        nn.init.xavier_normal_(tensor)

    return tensor

def rnn_init(tensor):
    if tensor.ndimension() != 2:
        return default_init(tensor)

    r, c = tensor.size()

    if r % c == 0:
        dim = 0
        n = r // c
        sub_size = (c, c)
    elif c % r == 0:
        dim = 1
        n = c // r
        sub_size = (r, r)
    else:
        return default_init(tensor)

    sub_tensors = [torch.Tensor(*sub_size).normal_(0, 1) for _ in range(n)]
    sub_tensors = [torch.svd(w, some=True)[0] for w in sub_tensors]

    init_tensor = torch.cat(sub_tensors, dim=dim)  # [r, c]

    with torch.no_grad():  # inplace op should be wrapped in no_grad mode.
        tensor.copy_(init_tensor)

    return tensor

def beam_search(encoded_code, desc_raw):
    tmp = desc_raw.split(' ')
    raw_len = len(tmp)
    new_len = raw_len * 0.4
    new_str = tmp[:int(new_len)]
    new_str = ' '.join(new_str)

    return new_str

def tile_batch(x, multiplier, batch_dim=0):
    x_size = x.size()
    out_size = x_size[:batch_dim] + (x_size[batch_dim] * multiplier,) + x_size[batch_dim + 1:]

    x_tiled = torch.unsqueeze(x, dim=batch_dim + 1)
    x_tiled = x_tiled.repeat(*[1 if d != batch_dim + 1 else multiplier for d in range(len(x_size) + 1)])
    x_tiled = x_tiled.view(*out_size)

    return x_tiled


_FLOAT32_INF = np.float32(np.finfo('float32').max / 10)

def mask_scores(scores, beam_mask, eos_id):
    """
    Mask scores of next step according to beam mask.
    Args:
        scores (torch.Tensor): Scores of next tokens with shape [batch_size, beam_size, vocab_size].
            Smaller should be better (usually negative log-probs).
        beam_mask (torch.Tensor): Mask of beam. 1.0 means not closed and vice verse. The shape is
            [batch_size, beam_size]

    Returns:
        Masked scores of next tokens.
    """
    vocab_size = scores.size(-1)

    finished_row = beam_mask.new(vocab_size, ).zero_() + float(_FLOAT32_INF)

    # If beam finished, only PAD could be generated afterwards.
    finished_row[eos_id] = 0.0

    scores = scores * beam_mask.unsqueeze(2) + \
             torch.matmul((1.0 - beam_mask).unsqueeze(2), finished_row.unsqueeze(0))

    return scores


def tensor_gather_helper(gather_indices,
                         gather_from,
                         batch_size,
                         beam_size,
                         gather_shape):
    range_ = (torch.arange(0, batch_size) * beam_size).long()

    range_ = range_.type_as(gather_indices)

    gather_indices_ = (gather_indices + torch.unsqueeze(range_, 1)).view(-1)

    output = torch.index_select(gather_from.view(*gather_shape), 0, gather_indices_)

    out_size = gather_from.size()[:1 + len(gather_shape)]

    return output.view(*out_size)


def reranking_beams(word_ids, scores, pad_id):
    word_ids = word_ids.cpu().numpy()
    scores = scores.cpu().numpy()

    # Reranking beams
    reranked_beams = np.argsort(scores, axis=1)
    reranked_word_ids = np.ones_like(word_ids) * pad_id

    for b in range(scores.shape[0]):
        for ii in reranked_beams[b]:
            reranked_word_ids[b, ii] = word_ids[b, ii]

    reranked_word_ids = reranked_word_ids.tolist()

    return reranked_word_ids


class LexicalState(object):
    def __init__(self, collrule, sub_tokens):
        self.collrule = collrule
        self.length = collrule.length
        self.count = collrule.count
        self.min_step = collrule.min_step
        self.tokens = sub_tokens  # just for correctness verification
        self.added_score = 0
        self.delta_score = {}

    def id_next(self):
        return self.collrule.id_next()

    def id_next_with_position(self):
        return self.collrule.id_next_with_position()