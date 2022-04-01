class EarlyStopping():
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.lowest_loss = float('inf')

    def check(self, loss):
                
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            else:
                return False