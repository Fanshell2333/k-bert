import torch
from src.k_bert.layers.embedding import BertEmbedding
from src.k_bert.encoders.bert_encoder import BertEncoder
from src.k_bert.targets.bert_target import BertTarget
from src.k_bert.targets.cls_target import ClsTarget
from src.k_bert.models.model import Model


def build_model(args):
    """
    Build universal encoder representations models.
    The combinations of different embedding, encoder,
    and targets layers yield pretrained models of different
    properties.
    We could select suitable one for downstream tasks.
    """

    embedding = BertEmbedding(args, len(args.vocab))
    encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
    target = globals()[args.target.capitalize() + "Target"](args, len(args.vocab))
    model = Model(args, embedding, encoder, target)
    return model
