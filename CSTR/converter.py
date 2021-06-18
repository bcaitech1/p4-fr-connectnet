import torch

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

class SRNConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, PAD=2):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        # list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        self.character = character
        self.PAD = PAD

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=254):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) for s in text]  # +1 for [s] at end of sentence.
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.cuda.LongTensor(len(text), batch_max_length).fill_(self.PAD)
        # mask_text = torch.cuda.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            # t_mask = [1 for i in range(len(text) + 1)]
            batch_text[i][0:len(t)] = torch.cuda.LongTensor(t)  # batch_text[:, len_text+1] = [EOS] token
            # mask_text[i][0:len(text)+1] = torch.cuda.LongTensor(t_mask)
        return (batch_text, torch.cuda.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            idx = text.find('$')
            texts.append(text[:idx])
        return texts

class FCConverter(object):

    def __init__(self, character, batch_max_length=254):
        list_character = list(character)
        self.batch_max_length = batch_max_length
        self.dict = dict()
        for i, char in enumerate(list_character):
            self.dict[char] = i
        self.ignore_index = self.dict['<PAD>']

    def encode(self, text):
        length = [len(s) for s in text]  # +1 for [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.ignore_index)  # noqa 501
        for i, t in enumerate(text):
            batch_text[i][:len(t)] = torch.LongTensor(t)
        batch_text_input = batch_text
        batch_text_target = batch_text

        return batch_text_input, torch.IntTensor(length), batch_text_target
