from classifier import Classifier
from dataGenerator import DataGenerator
import torch
import tiktoken

from lossCalculator import kl_divergence_loss, kl_loss_between_pair

tiktoker = tiktoken.get_encoding('gpt2')

dataGenerator = DataGenerator()
# tiktoker.max_token_value
classifier = Classifier(tokenSize=2, outputSize=4, embeddingSize=4, numOfHeads=1, tokenLegth=10)
optimizer = torch.optim.AdamW(classifier.parameters())
# this should be 10x4 (and another one to compare), could also be 10x10x4

for i in range(1000):
    samples, pairs = dataGenerator.generateSample(10, 10)  # 10 x 10 -> 10 x 10 x 4 -> 10 x 10 x 4
    optimizer.zero_grad()
    # outputs = classifier(torch.tensor(tiktoker.encode_ordinary_batch(samples)))
    outputs = classifier(torch.tensor(samples))
    loss = kl_divergence_loss(outputs, pairs)  # Compute loss
    loss.backward()
    optimizer.step()
    print(loss)


zeros = classifier(torch.tensor([[0]*10]))
ones = classifier(torch.tensor([[1]*10]))
mix = classifier(torch.tensor([[1,1,0,0,0,1,0,0,1,1]]))

print("loss between 0s")
print(kl_loss_between_pair(zeros, zeros))

print("loss between 1s")
print(kl_loss_between_pair(ones, ones))

print("loss between 0s and 1s")
print(kl_loss_between_pair(zeros, ones))

print("loss between 0s and mix")
print(kl_loss_between_pair(zeros, mix))

print("loss between 1s and mix")
print(kl_loss_between_pair(ones, mix))

print("loss between mix and mix")
print(kl_loss_between_pair(mix, mix))



# How, 2 similar lines...
# maximize likelihood. So values that are similar are maximized. Hence values are weighted to 1 or 0.
# but why 1 or 0 and to which ones.
# if similar then 1, even if low values.
# ok so its not about what values, but closer, the bigger jump.

# but end results? hmm. lets try
