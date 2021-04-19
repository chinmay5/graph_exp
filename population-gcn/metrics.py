import os

import torch
from sklearn import metrics


def masked_accuracy(preds, labels):
    """Accuracy with masking."""
    correct_prediction = torch.max(preds, dim=1)[1] == labels
    return correct_prediction.to(torch.float).mean()


class MaskedAUC(object):
    def __init__(self):
        self.filename = 'predictions.pth'
        self.is_finished = False

    def mark_finished(self):
        self.is_finished = True

    def evaluate_final(self):
        if not self.is_finished:
            raise ValueError("Please change the state once training is complete")
        data = torch.load(self.filename)
        preds = data['preds']
        labels = data['labels']
        return metrics.roc_auc_score(y_score=preds.cpu().numpy(), y_true=labels.cpu().numpy())

    def __del__(self):
        os.remove(self.filename)

    def masked_auc(self, preds, labels):
        if os.path.exists(self.filename):
            previous_data = torch.load(self.filename)
            previous_preds = previous_data['preds']
            previous_labels = previous_data['labels']
            preds = torch.cat((previous_preds, preds), dim=0)
            labels = torch.cat((previous_labels, labels), dim=0)
            data = {
                "preds": preds,
                "labels": labels
            }
            torch.save(data, self.filename)
        else:
            data = {
                "preds": preds,
                "labels": labels
            }
            torch.save(data, self.filename)


if __name__ == '__main__':
    auc_calc = MaskedAUC()
    for _ in range(5):
        preds, _ = torch.sort(torch.rand((5, )))
        labels = (2 * torch.rand((5, ))).to(torch.long)
        auc_calc.masked_auc(preds, labels)
    auc_calc.mark_finished()
    print(auc_calc.evaluate_final())
