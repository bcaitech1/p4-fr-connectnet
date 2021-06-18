import editdistance
import numpy as np

def word_error_rate(predicted_outputs, ground_truths):
    sum_wer=0.0
    for output,ground_truth in zip(predicted_outputs,ground_truths):
            output=output.split(" ")
            ground_truth=ground_truth.split(" ")
            distance = editdistance.eval(output, ground_truth)
            length = max(len(output),len(ground_truth))
            sum_wer+=(distance/length)
    return sum_wer/len(predicted_outputs)


def sentence_acc(predicted_outputs, ground_truths):
    correct_sentences=0
    for output,ground_truth in zip(predicted_outputs,ground_truths):
        if np.array_equal(output,ground_truth):
            correct_sentences+=1
    return correct_sentences/len(predicted_outputs)

def id_to_string(tokens, data_loader,do_eval=0):
    result = []
    if do_eval:
        special_ids = [data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
                       data_loader.dataset.token_to_id["<EOS>"]]

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != 2:
                        string += data_loader.dataset.id_to_token[token] + " "
                else:
                    if token == 1:
                        break
        else:
            for token in example:
                token = token.item()
                if token != 2:
                    string += data_loader.dataset.id_to_token[token] + " "

        result.append(string)
    return result