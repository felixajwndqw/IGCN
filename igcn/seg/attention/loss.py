import torch


class DAFLoss(torch.nn.Module):
    def __init__(self,
                 seg_criterion=torch.nn.BCEWithLogitsLoss(),
                 vec_criterion=torch.nn.MSELoss()):
        super().__init__()
        self.seg_criterion = seg_criterion
        self.vec_criterion = vec_criterion

    def forward(self, output, target):
        if not (type(output[0]) == tuple or type(output[0]) == list):
            return self.vec_criterion(output, target)
        (
            first_attention_vectors,
            second_attention_vectors,
            attention_in_encodings,
            attention_out_encodings,
            segmentations
        ) = output
        guided_losses = [
            self.vec_criterion(pre_a, a)
            for pre_a, a in zip(first_attention_vectors, second_attention_vectors)
        ]
        reconstruction_losses = [
            self.vec_criterion(in_enc, out_enc)
            for in_enc, out_enc in zip(attention_in_encodings, attention_out_encodings)
        ]
        seg_losses = [
            self.seg_criterion(seg, target)
            for seg in segmentations
        ]
        return sum(seg_losses) + .25 * sum(guided_losses) + .1 * sum(reconstruction_losses)
