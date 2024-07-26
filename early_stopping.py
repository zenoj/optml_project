import torch


def early_stopping_factory(patience=2):
    patience_counter = 0
    max_acc = 0
    def early_stopping_closure(acc):
        nonlocal patience_counter, max_acc
        if acc > max_acc:
            max_acc = acc
            patience_counter = 0
            return False
        else:
            patience_counter += 1
            if patience_counter > patience:
                return True
        return False
    return early_stopping_closure
