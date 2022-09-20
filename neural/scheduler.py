import pdb

"""
    Description:
        Has a patience and lowest value attribute
        Everytime check is called
            if value is not lowest recorded
                patience reset
                new lowest value = current value
            else
                patience -= 1

    Outputs:
        True (bool) if lowest value detected
        patience left (int) if not
"""
class EarlyStopping():
    def __init__(self, patience, verbose=True):
        self.patience = patience
        self.counter = 0
        self.lowest_loss = float('inf')
        self.verbose = verbose

    def check(self, loss):
                
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            self.counter = 0
            if self.verbose:
                print(f'Early stopping: New lowest loss {loss} logged. Patience reset.')
                return 'lowest_loss'
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'Early Stopping: Patience threshold exceeded!')
                return 0
            else:
                print(f'Early stopping: Loss {loss} is not the lowest. Patience now at {self.patience-self.counter}.')
                return self.patience-self.counter