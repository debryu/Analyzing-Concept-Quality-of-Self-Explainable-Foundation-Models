import torch 

# Get the value of sigmoid(6)
def sigmoid_6():
    return torch.sigmoid(torch.tensor(6.0))

print(sigmoid_6().item()) # = 0.99752


truth_value = [1.0, 0.0]
predicted_value = [0.3, 0.7]

def compute_interval(preds):
    ins = predicted_value[:0]
# Calculate the binary cross entropy loss
def binary_cross_entropy_loss():
    return torch.nn.BCELoss()(torch.tensor(predicted_value), torch.tensor(truth_value))

print(binary_cross_entropy_loss().item()) # = 0.0003
# Generate a random number from a normal distribution with std 2
test_term = torch.randn_like(torch.tensor(predicted_value)) * 0.01
print(test_term)