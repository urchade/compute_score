import torch


def ex6(logits, activation_function, threshold, targets):
    """
    Parameters
    ----------
    logits: torch.Tensor of shape (n_samples,)
        logits contains the output of the NN before
        the application of activation_function.

    activation_function:
        The activation function to apply to logits

    threshold: torch.Tensor
        The threshold to decide if a sample is
        classified as True or False as

    targets: torch.Tensor of shape (n_samples,) and datatype torch.bool
        contains the true labels (=target classes) of the samples

    Returns
    -------
        true positive rate, true negative rate, false positive rate,
        false negative rate, accuracy, balanced accuracy
    """

    if logits.shape != targets.shape or targets.sum() == 0 or targets.sum() == targets.shape[0]:
        raise ValueError

    if not isinstance(logits, torch.FloatTensor) or not isinstance(threshold, torch.Tensor) \
            or not isinstance(targets, torch.BoolTensor):
        raise TypeError

    y = activation_function(logits) > threshold

    true_positives = ((targets == y) * (targets == True)).double().sum()
    true_negative = ((targets == y) * (targets == False)).double().sum()
    false_negative = ((targets != y) * (targets == True)).double().sum()
    false_positive = ((targets != y) * (targets == False)).double().sum()

    tpr = true_positives / (true_positives + false_negative)
    tnr = true_negative / (true_negative + false_positive)
    fnr = 1 - tpr
    fpr = 1 - tnr
    bacc = (tpr + tnr) / 2.

    accuracy = (targets == y).double().mean()

    return tpr.item(), tnr.item(), fpr.item(), fnr.item(), accuracy.item(), bacc.item()
