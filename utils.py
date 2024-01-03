import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataset import PaddedTensorDataset

def vectorized_data(data, item2id):
	# data is a list of tuples of the type ('Hori','Japanese')
	# item2id is defaultdict of all characters
	# return for each word its correspondence list
	# for instance 'jowett' -- > [17,27,43,30,38,38]

	return [[item2id[token] if token in item2id else item2id['UNK'] for token in seq] 
		 for seq, _ in data]


def pad_sequences(vectorized_seqs, seq_lengths):
	# Put each vectorized word in a 2d tensor (each word in a single row)
	# the remaining values in the row are 0

	# create a zero matrix
	seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()

	# fill the index
	for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
		seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
	return seq_tensor


def create_dataset(data, input2id, target2id, batch_size=4):
	#target2id is the defaultdict of languages
	#input2id is the defaultdict of characters
	vectorized_seqs = vectorized_data(data, input2id)

	seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])
	seq_tensor = pad_sequences(vectorized_seqs, seq_lengths)

	# extract labels and train data from data
	target_tensor = torch.LongTensor([target2id[y] for _, y in data])
	raw_data = [x for x, _ in data]

	return DataLoader(PaddedTensorDataset(seq_tensor, target_tensor, seq_lengths, raw_data), 
				   batch_size=batch_size)


def sort_batch(batch, targets, lengths):
	seq_lengths, perm_idx = lengths.sort(0, descending=True)
	seq_tensor = batch[perm_idx]
	target_tensor = targets[perm_idx]

	# just permutate in descending (lengths) order the batch
	return seq_tensor.transpose(0, 1), target_tensor, seq_lengths