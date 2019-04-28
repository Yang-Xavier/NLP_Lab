import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = "The mathematician ran. The mathematician ran to the store. The physicist ran to the store. The philosopher thought about it. The mathematician solved the open problem".split('.')

train_sent = []
trigrams = []
for sent in test_sentence:
    sent = ("<s> " + sent.strip() + " . </s>").split()
    trg = [([sent[i], sent[i + 1]], sent[i + 2]) for i in range(len(sent) - 2)]
    trigrams.extend(trg)
    train_sent.extend(sent)

vocab = train_sent
word_to_ix = {word: i for i, word in zip(range(0, len(vocab)), vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear1(embeds)
        out = F.relu(out)
        out = self.linear2(out)

        log_probs = F.log_softmax(out, dim=1)

        return log_probs

def prediction(context):
    context_idxs = [word_to_ix[w] for w in context]
    idxs = autograd.Variable(torch.LongTensor(context_idxs))
    log_probs=model(idxs)
    return torch.argmax(log_probs), log_probs

losses = []
loss_function = nn.NLLLoss()

def valid(model) :
    validate_sentence = "The mathematician ran to the store ."
    sent = ("<s> " + validate_sentence.strip() + " </s>").split()
    print("\nCHECK \nSTART WITH: ", sent[:2])
    for i in range(2, len(sent)):
        print(vocab[prediction([sent[i-2], sent[i-1]])[0]], end="  ")

    print("\nTEST")
    test_sent = "The _ solved the open problem ."
    sent =  ("<s> " + test_sent.strip() + " </s>").split()
    options = ["physicist", "philosopher"]
    q_idx = sent.index('_')
    probs = []
    for op in options:
        p1 = prediction([sent[q_idx-2], sent[q_idx-1]])[1][:,word_to_ix[op]] # <s> The _
        p2 = prediction([sent[q_idx-1], op])[1][:,word_to_ix[sent[q_idx+1]]] # The _ solve
        p3 = prediction([op,sent[q_idx+1]])[1][:,word_to_ix[sent[q_idx+2]]] # _ solve the
        probs.append(p1+p2+p3)

    # F.cosine_similarity(input1, input2)
    print("Question: ", test_sent)
    print("Options: ", options)
    print("Answer: ", options[probs.index(max(probs))], max(probs))
    m_idx = autograd.Variable(torch.LongTensor([word_to_ix["mathematician"]]))
    phi_idx = autograd.Variable(torch.LongTensor([word_to_ix[options[0]]]))
    phy_idx = autograd.Variable(torch.LongTensor([word_to_ix[options[1]]]))

    print("Similarity between mathematician and ", options[0], ":", abs(F.cosine_similarity(model.embeddings(m_idx), model.embeddings(phi_idx))))
    print("Similarity between mathematician and ", options[1], ":", abs(F.cosine_similarity(model.embeddings(m_idx), model.embeddings(phy_idx))))
    print("Similarity between", options[1], "and",options[0], ":", abs(F.cosine_similarity(model.embeddings(phi_idx), model.embeddings(phy_idx))))

print("_____epoch 10, learning rate 0.001, embedding dimension 10_____")
EMBEDDING_DIM = 10
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
losses = []
for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        model.zero_grad()
        log_probs = prediction(context)[1]
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    losses.append(total_loss)
print("Loss: ", losses)  # The loss decreased every iteration over the training data!
valid(model)

print("\n\n_____epoch 10, learning rate 0.03, embedding dimension 10_____")
EMBEDDING_DIM = 10
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.03)
losses = []
for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        model.zero_grad()
        log_probs = prediction(context)[1]
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    losses.append(total_loss)
print("Loss: ", losses)  # The loss decreased every iteration over the training data!
valid(model)

print("\n\n_____epoch 20, learning rate 0.03, embedding dimension 10_____")
EMBEDDING_DIM = 10
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.03)
losses = []
for epoch in range(20):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        model.zero_grad()
        log_probs = prediction(context)[1]
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    losses.append(total_loss)
print("Loss: ", losses)  # The loss decreased every iteration over the training data!
valid(model)

print("\n\n_____epoch 100, learning rate 0.001, embedding dimension 10_____")
EMBEDDING_DIM = 10
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
losses = []
for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        model.zero_grad()
        log_probs = prediction(context)[1]
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    losses.append(total_loss)
print("Loss: ", losses)  # The loss decreased every iteration over the training data!
valid(model)

print("\n\n_____epoch 20, learning rate 0.03, embedding dimension 20_____")
EMBEDDING_DIM = 20
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.03)
losses = []
for epoch in range(20):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        model.zero_grad()
        log_probs = prediction(context)[1]
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    losses.append(total_loss)
print("Loss: ", losses)  # The loss decreased every iteration over the training data!
valid(model)