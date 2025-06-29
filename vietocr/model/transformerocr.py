from vietocr.model.backbone.cnn import CNN
from vietocr.model.seqmodel.transformer import LanguageTransformer
from vietocr.model.seqmodel.seq2seq import Seq2Seq
from vietocr.model.seqmodel.convseq2seq import ConvSeq2Seq
from torch import nn


class VietOCR(nn.Module):
    def __init__(
        self,
        vocab_size,
        backbone,
        cnn_args,
        transformer_args,
        seq_modeling="transformer",
    ):

        super(VietOCR, self).__init__()

        self.cnn = CNN(backbone, **cnn_args)
        self.seq_modeling = seq_modeling

        if seq_modeling == "transformer":
            self.transformer = LanguageTransformer(vocab_size, **transformer_args)
        elif seq_modeling == "seq2seq":
            self.transformer = Seq2Seq(vocab_size, **transformer_args)
        elif seq_modeling == "convseq2seq":
            self.transformer = ConvSeq2Seq(vocab_size, **transformer_args)
        else:
            raise ("Not Support Seq Model")

    def forward(self, batch):
        # Giải nén dictionary bên trong hàm forward
        img = batch['img']
        tgt_input = batch['tgt_input']
        # Dùng .get() để an toàn nếu key không tồn tại
        tgt_key_padding_mask = batch.get('tgt_padding_mask', None)
        src = self.cnn(img)
        src = src.permute(1, 0, 2)
        if self.seq_modeling == "transformer":
            outputs = self.transformer(
                src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask
            )
        elif self.seq_modeling == "seq2seq":
            outputs = self.transformer(src, tgt_input)
        elif self.seq_modeling == "convseq2seq":
            outputs = self.transformer(src, tgt_input)
        return outputs
