import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset


# 학습 or 평가 데이터를 읽어 리스트에 저장
def read_data(file_path, mode):
    with open(file_path, "r", encoding="utf8") as inFile:
        lines = inFile.readlines()

    datas = []
    for index, line in enumerate(tqdm(lines, desc="read_data")):
        # 입력 문장을 \t으로 분리
        pieces = line.strip().split("\t")

        if mode == "train" or mode == "test":
            # 데이터의 형태가 올바른지 체크
            assert len(pieces) == 3
            assert len(pieces[0].split(" ")) == len(pieces[2].split(" "))
            sentence, senti_label, score_label = pieces[0], int(pieces[1]), [int(score) + 1 for score in pieces[2].split(" ")]
            datas.append((sentence, senti_label, score_label))
        elif mode == "analyze":
            sentence, senti_label = pieces[0], int(pieces[1])
            datas.append((sentence, senti_label))
    return datas


def convert_data2dataset(datas, tokenizer, max_length, labels, score_labels, mode):
    total_input_ids, total_attention_mask, total_token_type_ids, total_senti_labels, total_score_labels, total_senti_seq, total_score_seq, total_word_seq = [], [], [], [], [], [], [], []

    if mode == "analyze":
        total_score_labels = None
    for index, data in enumerate(tqdm(datas, desc="convert_data2dataset")):
        sentence = ""
        senti_label = 0
        score_label = []
        if mode == "train" or mode == "test":
            sentence, senti_label, score_label = data
        elif mode == "analyze":
            sentence, senti_label = data

        # tokens = tokenizer.tokenize(sentence)

        # tokens = ["[CLS]"] + tokens
        # tokens = tokens[:max_length-1]
        # tokens.append("[SEP]")
        input_ids = [tokenizer._convert_token_to_id(token) for token in sentence.split()]
        assert len(input_ids) <= max_length

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding = [0] * (max_length - len(input_ids))

        total_word_seq.append(len(input_ids))

        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_token_type_ids.append(token_type_ids)
        total_senti_labels.append(senti_label)

        total_senti_seq.append([i for i in range(labels)])
        total_score_seq.append([i for i in range(score_labels)])

        if mode == "train" or mode == "test":
            score_label += padding
            total_score_labels.append(score_label)

        if index < 2:
            print("*** Example ***")
            print("sequence : {}".format(sentence))
            print("input_ids: {}".format(" ".join([str(x) for x in total_input_ids[-1]])))
            print("attention_mask: {}".format(" ".join([str(x) for x in total_attention_mask[-1]])))
            print("token_type_ids: {}".format(" ".join([str(x) for x in total_token_type_ids[-1]])))
            print("label: {}".format(total_senti_labels[-1]))
            print("senti_seq: {}".format(total_senti_seq[-1]))
            print("score_seq: {}".format(total_score_seq[-1]))
            print("word_seq: {}".format(total_word_seq[-1]))
            print()
            if mode == "train" or mode == "test":
                print("score label: {}".format(total_score_labels[-1]))

    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)
    total_senti_labels = torch.tensor(total_senti_labels, dtype=torch.long)
    total_senti_seq = torch.tensor(total_senti_seq, dtype=torch.long)
    total_score_seq = torch.tensor(total_score_seq, dtype=torch.long)
    total_word_seq = torch.tensor(total_word_seq, dtype=torch.long)
    if mode == "train" or mode == "test":
        total_score_labels = torch.tensor(total_score_labels, dtype=torch.long)
        dataset = TensorDataset(total_input_ids, total_attention_mask, total_token_type_ids, total_senti_labels,
                            total_score_labels, total_senti_seq, total_score_seq, total_word_seq)
    elif mode == "analyze":
        dataset = TensorDataset(total_input_ids, total_attention_mask, total_token_type_ids, total_senti_labels,
                                total_senti_seq, total_score_seq, total_word_seq)

    return dataset
