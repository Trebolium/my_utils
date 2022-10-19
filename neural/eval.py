import torch


# generate classification loss using predictions and labels
def get_accuracy(self, predictions, y_data):
    _, predicted = torch.max(predictions.data, 1)
    correct_preds = (predicted == y_data).sum().item()
    return correct_preds / len(y_data) #accuracy