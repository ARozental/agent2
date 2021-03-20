from src.model import AgentModel
from src.tree_dataset import TreeDataset
from src.pre_processing import TreeTokenizer
import torch


from src.utils import seed_torch
seed_torch(0) #learns faster than 777


USE_CUDA = False

dataset = TreeDataset()
device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
model = AgentModel(TreeTokenizer())
model.to(device)
# model.train()
model.eval()
for batch_tree in dataset.iterator():
    model.forward(batch_tree)
    break

main_params = [param for name,param in model.named_parameters() if ("discriminator" not in name) and ("generator" not in name)]
generator_params = [param for name,param in model.named_parameters() if "generator" in name]
discriminator_params = [param for name,param in model.named_parameters() if "discriminator" in name]

main_optimizer = torch.optim.Adam(main_params, 0.002)
generator_optimizer = torch.optim.Adam(generator_params, 0.002)
discriminator_optimizer = torch.optim.Adam(discriminator_params, 0.002)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)
#optimizer_D = torch.optim.AdamW(D.parameters(), 0.001)
#d_loss.backward(retain_graph=True)

for epoch in range(10001):
    # print('Epoch', epoch + 1)
    generate = True

    for batch in dataset.iterator():
        model.train()
        main_optimizer.zero_grad()

        g_loss, disc_loss, main_loss, loss_object  = model.forward(batch,generate=generate,epoch=epoch)
        if generate == True:
            disc_loss.backward(retain_graph=True)
            (100*g_loss-disc_loss).backward(retain_graph=True)
            generator_optimizer.step()
            discriminator_optimizer.step()

        main_loss.backward()
        main_optimizer.step()


        # if epoch % 100 == 0:
        #     print("epoch:",epoch,"total_loss:",main_loss.item(),"loss object:",loss_object)
        #     words = model.debug_decode(batch).detach().numpy()
        #     pred = [dataset.tree_tokenizer.detokenize(w) for w in words]
        #     print(pred)

            # model.eval()
            # print('Word Level')
            # print(batch)
            # exit()
            # examples = dataset.debug_examples(level=0)
            # word_vecs = model.encode(examples)
            # preds = model.debug_decode(word_vecs, level=0)
            # for pred, word in zip(preds, examples):
            #     print(dataset.decode(pred), end='')
            #     if dataset.decode(pred) == dataset.decode(word):
            #         print('   MATCHED!', end='')
            #     print('')
            #
            # if NUM_LEVELS > 1:
            #     print('Sentence Level')
            #     examples = dataset.debug_examples(level=1)
            #     sent_vec = model.encode(examples)
            #     preds = model.debug_decode(sent_vec, level=1)
            #     preds = preds.cpu().detach().numpy()
            #     for pred, sent in zip(preds, examples):
            #         print(dataset.decode(pred), end='')
            #         if dataset.decode(pred) == dataset.decode(sent):
            #             print('   MATCHED!', end='')
            #         print('')
            #
            #     # Word Vector distance
            #     word_vectors = [model.encode(inputs[i], mask[i], level=0) for i in range(len(sentences))]
            #     word_vecs_from_sentences = model.decode(sent_vec, level=1, return_word_vectors=True)
            #     for word_vecs, word_from_sent in zip(word_vectors, word_vecs_from_sentences):
            #         # print(word_vecs[:, :5])  # This shows the first 5 values of each word vector
            #         print('Distances:', torch.norm(word_vecs - word_from_sent, dim=1).detach().numpy())
