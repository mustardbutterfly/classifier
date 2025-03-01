import torch


class Classifier(torch.nn.Module):
    def __init__(self, tokenSize, embeddingSize, outputSize, numOfHeads, tokenLegth):
        super(Classifier, self).__init__()
        self.key = torch.nn.Linear(embeddingSize, embeddingSize)
        self.query = torch.nn.Linear(embeddingSize, embeddingSize)
        self.value = torch.nn.Linear(embeddingSize, embeddingSize)
        self.multiAttentionHead = torch.nn.MultiheadAttention(embeddingSize, numOfHeads)
        self.proj = torch.nn.Linear(embeddingSize, embeddingSize)
        self.normInput = torch.nn.LayerNorm(embeddingSize)
        self.normOutput = torch.nn.LayerNorm(embeddingSize)
        self.norm = torch.nn.LayerNorm(embeddingSize)
        self.out = torch.nn.Linear(embeddingSize, outputSize)
        self.input = torch.nn.Embedding(tokenSize, embeddingSize)
        self.posEmbedding = torch.nn.Embedding(tokenLegth, embeddingSize)

    def forward(self, x):
        x = self.normInput(self.input(x) + self.posEmbedding(torch.arange(x.size(1))))
        keyX = self.key(x)
        queryX = self.query(x)
        valueX = self.value(x)
        attn_output = self.multiAttentionHead(queryX, keyX, valueX, need_weights=False)[0]
        norm_attn_output = self.norm(attn_output)
        return self.out(self.normOutput(x + self.proj(norm_attn_output)))
