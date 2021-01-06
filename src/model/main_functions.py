import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from transformers.configuration_electra import ElectraConfig
from transformers.tokenization_electra import ElectraTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from src.model.model import ElectraForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from src.functions import preprocessing
from torch.utils.data import TensorDataset
import os


class Helper():
    def __init__(self, config):
        self.config = config

    def do_train(self, electra_model, optimizer, scheduler, train_dataloader, epoch, global_step):
        score_criterion = nn.CrossEntropyLoss(ignore_index=0)
        senti_criterion = nn.CrossEntropyLoss()

        # batch 단위 별 loss를 담을 리스트
        losses = []
        # 모델의 출력 결과와 실제 정답값을 담을 리스트
        total_predicts, total_corrects = [], []
        total_pred_scores, total_score_corrects = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="do_train(epoch_{})".format(epoch))):

            batch = tuple(t.cuda() for t in batch)
            input_ids, attention_mask, token_type_ids, senti_labels, score_labels, senti_seq, score_seq, word_len_seq = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]

            # 입력 데이터에 대한 출력과 loss 생성
            score_logits, senti_logit = electra_model(input_ids, attention_mask, token_type_ids,
                                                                        senti_labels, score_seq, senti_seq,
                                                                        word_len_seq)

            score_loss = score_criterion(score_logits, score_labels)

            score_pred = F.softmax(score_logits, dim=1)
            score_pred = score_pred.argmax(dim=1)

            for pred, gold, length in zip(score_pred, score_labels, word_len_seq):
                pred = pred[:length]
                gold = gold[:length]
                for pred_, gold_ in zip(pred, gold):
                    if gold_ != 0:
                        if pred_ == gold_:
                            total_score_corrects += 1
                        total_pred_scores += 1

            senti_logit = senti_logit.squeeze()
            senti_loss = senti_criterion(senti_logit, senti_labels)

            total_loss = score_loss * 0.1 + senti_loss * 0.9
            # total_loss = senti_loss

            predicts = F.softmax(senti_logit, dim=1)
            predicts = predicts.argmax(dim=-1)
            predicts = predicts.cpu().detach().numpy().tolist()
            labels = senti_labels.cpu().detach().numpy().tolist()

            total_predicts += predicts
            total_corrects += labels

            if self.config["gradient_accumulation_steps"] > 1:
                total_loss = total_loss / self.config["gradient_accumulation_steps"]
            if step % 100 == 0:
                print("loss : ", '{:.6f}'.format(total_loss))

            # loss 값으로부터 모델 내부 각 매개변수에 대하여 gradient 계산
            total_loss.backward()
            losses.append(total_loss.data.item())

            if (step + 1) % self.config["gradient_accumulation_steps"] == 0 or \
                    (len(train_dataloader) <= self.config["gradient_accumulation_steps"] and (step + 1) == len(
                        train_dataloader)):
                torch.nn.utils.clip_grad_norm_(electra_model.parameters(), self.config["max_grad_norm"])

                # 모델 내부 각 매개변수 가중치 갱신
                optimizer.step()
                scheduler.step()

                # 변화도를 0으로 변경
                electra_model.zero_grad()
                global_step += 1

        # 정확도 계산
        accuracy = accuracy_score(total_corrects, total_predicts)

        score_acc = total_score_corrects / total_pred_scores

        return accuracy, np.mean(losses), global_step, score_acc

    def do_evaluate(self, electra_model, test_dataloader, mode):
        # 모델의 입력, 출력, 실제 정답값을 담을 리스트
        total_input_ids, total_predicts, total_corrects = [], [], []
        total_pred_scores, total_score_corrects = 0, 0
        for step, batch in enumerate(tqdm(test_dataloader, desc="do_evaluate")):
            batch = tuple(t.cuda() for t in batch)
            input_ids, attention_mask, token_type_ids, senti_labels, score_labels, senti_seq, score_seq, word_len_seq = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]

            # 입력 데이터에 대한 출력 결과 생성
            score_logits, senti_logits = electra_model(input_ids, attention_mask, token_type_ids,
                                                                         senti_labels, score_seq, senti_seq,
                                                                         word_len_seq)

            score_pred = F.softmax(score_logits, dim=1)
            score_pred = score_pred.argmax(dim=1)

            for pred, gold, length in zip(score_pred, score_labels, word_len_seq):
                pred = pred[:length]
                gold = gold[:length]
                for pred_, gold_ in zip(pred, gold):
                    if gold_ != 0:
                        if pred_ == gold_:
                            total_score_corrects += 1
                        total_pred_scores += 1

            senti_logits = senti_logits.squeeze()

            predicts = F.softmax(senti_logits, dim=1)
            predicts = predicts.argmax(dim=-1)
            predicts = predicts.cpu().detach().numpy().tolist()
            labels = senti_labels.cpu().detach().numpy().tolist()
            input_ids = input_ids.cpu().detach().numpy().tolist()

            total_predicts += predicts
            total_corrects += labels
            total_input_ids += input_ids

        # 정확도 계산
        accuracy = accuracy_score(total_corrects, total_predicts)
        score_acc = total_score_corrects / total_pred_scores

        if (mode == "train"):
            return accuracy, score_acc
        else:
            return accuracy, total_input_ids, total_predicts, total_corrects

    def do_analyze(self, electra_model, test_dataloader, mode):
        # 모델의 입력, 출력, 실제 정답값을 담을 리스트
        total_input_ids, total_predicts, total_corrects = [], [], []

        for step, batch in enumerate(tqdm(test_dataloader, desc="do_analyze")):
            batch = tuple(t.cuda() for t in batch)
            input_ids, attention_mask, token_type_ids, senti_labels, senti_seq, score_seq, word_len_seq = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

            # 입력 데이터에 대한 출력 결과 생성
            score_logits, senti_logits = electra_model(input_ids, attention_mask, token_type_ids,
                                                                         senti_labels, score_seq, senti_seq,
                                                                         word_len_seq)

            senti_logits = senti_logits.squeeze()

            predicts = F.softmax(senti_logits, dim=1)
            predicts = predicts.argmax(dim=-1)
            predicts = predicts.cpu().detach().numpy().tolist()
            labels = senti_labels.cpu().detach().numpy().tolist()
            input_ids = input_ids.cpu().detach().numpy().tolist()

            total_predicts += predicts
            total_corrects += labels
            total_input_ids += input_ids

        # 정확도 계산
        accuracy = accuracy_score(total_corrects, total_predicts)
        return accuracy, total_input_ids, total_predicts, total_corrects

    def train(self):
        #########################################################################################################################################
        # electra config 객체 생성
        electra_config = ElectraConfig.from_pretrained(os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
                                                       num_labels=self.config["senti_labels"], cache_dir=None)

        # electra tokenizer 객체 생성
        electra_tokenizer = ElectraTokenizer.from_pretrained(os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
                                                             do_lower_case=False,
                                                             cache_dir=None)

        # electra model 객체 생성
        electra_model = ElectraForSequenceClassification.from_pretrained(os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
                                                                         config=electra_config,
                                                                         lstm_hidden=self.config['lstm_hidden'],
                                                                         label_emb_size=self.config['lstm_hidden'] * 2,
                                                                         score_emb_size=self.config['lstm_hidden'] * 2,
                                                                         score_size=self.config['score_labels'],
                                                                         num_layer=self.config['lstm_num_layer'],
                                                                         bilstm_flag=self.config['bidirectional_flag'],
                                                                         cache_dir=self.config["cache_dir_path"]
                                                                         # from_tf=True
                                                                         )
        #########################################################################################################################################

        electra_model.cuda()

        # 학습 데이터 읽기
        train_datas = preprocessing.read_data(file_path=self.config["train_data_path"], mode=self.config["mode"])

        # 학습 데이터 전처리
        train_dataset = preprocessing.convert_data2dataset(datas=train_datas, tokenizer=electra_tokenizer,
                                                           max_length=self.config["max_length"],
                                                           labels=self.config["senti_labels"],
                                                           score_labels=self.config["score_labels"],
                                                           mode=self.config["mode"])

        # 학습 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.config["batch_size"])

        # 평가 데이터 읽기
        test_datas = preprocessing.read_data(file_path=self.config["test_data_path"], mode=self.config["mode"])

        # 평가 데이터 전처리
        test_dataset = preprocessing.convert_data2dataset(datas=test_datas, tokenizer=electra_tokenizer,
                                                          max_length=self.config["max_length"],
                                                          labels=self.config["senti_labels"],
                                                          score_labels=self.config["score_labels"],
                                                          mode=self.config["mode"])

        # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)

        # 전체 학습 횟수(batch 단위)
        t_total = len(train_dataloader) // self.config["gradient_accumulation_steps"] * self.config["epoch"]

        # 모델 학습을 위한 optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer = AdamW([{'params': [p for n, p in electra_model.named_parameters() if not any(nd in n for nd in no_decay)],
                            'lr': 5e-5, 'weight_decay': self.config['weight_decay']},
                           {'params': [p for n, p in electra_model.named_parameters() if any(nd in n for nd in no_decay)],
                            'lr': 5e-5, 'weight_decay': 0.0}])
        # optimizer = AdamW(lan.parameters(), lr=self.config['learning_rate'], eps=self.config['adam_epsilon'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config["warmup_steps"],
                                                    num_training_steps=t_total)

        if os.path.isfile(os.path.join(self.config["model_dir_path"], "optimizer.pt")) and os.path.isfile(
                os.path.join(self.config["model_dir_path"], "scheduler.pt")):
            # 기존에 학습했던 optimizer와 scheduler의 정보 불러옴
            optimizer.load_state_dict(torch.load(os.path.join(self.config["model_dir_path"], "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.config["model_dir_path"], "scheduler.pt")))
            print("#######################     Success Load Model     ###########################")

        global_step = 0
        electra_model.zero_grad()
        max_test_accuracy = 0
        for epoch in range(self.config["epoch"]):
            electra_model.train()

            # 학습 데이터에 대한 정확도와 평균 loss
            train_accuracy, average_loss, global_step, score_acc = self.do_train(electra_model=electra_model,
                                                                 optimizer=optimizer, scheduler=scheduler,
                                                                 train_dataloader=train_dataloader,
                                                                 epoch=epoch + 1, global_step=global_step)

            print("train_accuracy : {}\taverage_loss : {}\n".format(round(train_accuracy, 4), round(average_loss, 4)))
            print("train_score_accuracy :", "{:.6f}".format(score_acc))

            electra_model.eval()

            # 평가 데이터에 대한 정확도
            test_accuracy, score_acc = self.do_evaluate(electra_model=electra_model, test_dataloader=test_dataloader,
                                             mode=self.config["mode"])

            print("test_accuracy : {}\n".format(round(test_accuracy, 4)))
            print("test_score_accuracy :", "{:.6f}".format(score_acc))


            # 현재의 정확도가 기존 정확도보다 높은 경우 모델 파일 저장
            if (max_test_accuracy < test_accuracy):
                max_test_accuracy = test_accuracy

                output_dir = os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                electra_config.save_pretrained(output_dir)
                electra_tokenizer.save_pretrained(output_dir)
                electra_model.save_pretrained(output_dir)
                # torch.save(lan.state_dict(), os.path.join(output_dir, "lan.pt"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            print("max_test_accuracy :", "{:.6f}".format(round(max_test_accuracy, 4)))

    def show_result(self, total_input_ids, total_predicts, total_corrects, tokenizer):
        for index, input_ids in enumerate(total_input_ids):
            tokens = [tokenizer._convert_id_to_token(input_id) for input_id in input_ids]

            # [CLS] 토큰 제거
            tokens = tokens[1:]

            # [SEP] 토큰 제거
            tokens = tokens[:tokens.index("[SEP]")]

            # 입력 sequence 복원
            sequence = tokenizer.convert_tokens_to_string(tokens)

            predict, correct = total_predicts[index], total_corrects[index]
            if (predict == 0):
                predict = "negative"
            else:
                predict = "positive"

            if (correct == 0):
                correct = "negative"
            else:
                correct = "positive"

            print("sequence : {}".format(sequence))
            print("predict : {}".format(predict))
            print("correct : {}".format(correct))
            print()

    def test(self):
        # electra config 객체 생성
        electra_config = ElectraConfig.from_pretrained(
            os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
            num_labels=self.config["senti_labels"],
            cache_dir=None)

        # electra tokenizer 객체 생성
        electra_tokenizer = ElectraTokenizer.from_pretrained(
            os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
            do_lower_case=False,
            cache_dir=None)

        # electra model 객체 생성
        electra_model = ElectraForSequenceClassification.from_pretrained(
            os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
            config=electra_config,
            lstm_hidden=self.config['lstm_hidden'],
            label_emb_size=self.config['lstm_hidden'] * 2,
            score_emb_size=self.config['lstm_hidden'] * 2,
            score_size=self.config['score_labels'],
            num_layer=self.config['lstm_num_layer'],
            bilstm_flag=self.config['bidirectional_flag'],
            cache_dir=self.config["cache_dir_path"]
        )

        electra_model.cuda()

        # 평가 데이터 읽기
        test_datas = preprocessing.read_data(file_path=self.config["test_data_path"], mode=self.config["mode"])

        # 평가 데이터 전처리
        test_dataset = preprocessing.convert_data2dataset(datas=test_datas, tokenizer=electra_tokenizer,
                                                          max_length=self.config["max_length"],
                                                          labels=self.config["senti_labels"],
                                                          score_labels=self.config["score_labels"],
                                                          mode=self.config["mode"])

        # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)

        electra_model.eval()

        # 평가 데이터에 대한 정확도와 모델의 입력, 출력, 정답
        test_accuracy, total_input_ids, total_predicts, total_corrects = self.do_evaluate(electra_model=electra_model,
                                                                                     test_dataloader=test_dataloader,
                                                                                     mode=self.config["mode"])

        print("test_accuracy : {}\n".format(round(test_accuracy, 4)))

        # 10개의 평가 케이스에 대하여 모델 출력과 정답 비교
        self.show_result(total_input_ids=total_input_ids[:10], total_predicts=total_predicts[:10],
                    total_corrects=total_corrects[:10], tokenizer=electra_tokenizer)

    def analyze(self):
        # electra config 객체 생성
        electra_config = ElectraConfig.from_pretrained(
            os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
            num_labels=self.config["senti_labels"],
            cache_dir=None)

        # electra tokenizer 객체 생성
        electra_tokenizer = ElectraTokenizer.from_pretrained(
            os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
            do_lower_case=False,
            cache_dir=None)

        # electra model 객체 생성
        electra_model = ElectraForSequenceClassification.from_pretrained(
            os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
            config=electra_config,
            lstm_hidden=self.config['lstm_hidden'],
            label_emb_size=self.config['lstm_hidden'] * 2,
            score_emb_size=self.config['lstm_hidden'] * 2,
            score_size=self.config['score_labels'],
            num_layer=self.config['lstm_num_layer'],
            bilstm_flag=self.config['bidirectional_flag'],
            cache_dir=self.config["cache_dir_path"]
        )

        electra_model.cuda()

        # 평가 데이터 읽기
        test_datas = preprocessing.read_data(file_path=self.config["analyze_data_path"], mode=self.config["mode"])

        # 평가 데이터 전처리
        test_dataset = preprocessing.convert_data2dataset(datas=test_datas, tokenizer=electra_tokenizer,
                                                          max_length=self.config["max_length"],
                                                          labels=self.config["senti_labels"],
                                                          score_labels=self.config["score_labels"],
                                                          mode=self.config["mode"])

        # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)

        electra_model.eval()

        # 평가 데이터에 대한 정확도와 모델의 입력, 출력, 정답
        test_accuracy, total_input_ids, total_predicts, total_corrects = self.do_analyze(electra_model=electra_model,
                                                                                     test_dataloader=test_dataloader,
                                                                                     mode=self.config["mode"])

        print("test_accuracy : {}\n".format(round(test_accuracy, 4)))

        print("테스트 데이터 10개에 대하여 모델 출력과 정답을 비교")
        # 10개의 평가 케이스에 대하여 모델 출력과 정답 비교
        self.show_result(total_input_ids=total_input_ids[:10], total_predicts=total_predicts[:10],
                    total_corrects=total_corrects[:10], tokenizer=electra_tokenizer)

    def demo(self):
        # electra config 객체 생성
        electra_config = ElectraConfig.from_pretrained(
            os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
            num_labels=self.config["senti_labels"],
            cache_dir=None)

        # electra tokenizer 객체 생성
        electra_tokenizer = ElectraTokenizer.from_pretrained(
            os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
            do_lower_case=False,
            cache_dir=None)

        # electra model 객체 생성
        electra_model = ElectraForSequenceClassification.from_pretrained(
            os.path.join(self.config["model_dir_path"], "checkpoint-{}".format(self.config["checkpoint"])),
            config=electra_config,
            lstm_hidden=self.config['lstm_hidden'],
            label_emb_size=self.config['lstm_hidden'] * 2,
            score_emb_size=self.config['lstm_hidden'] * 2,
            score_size=self.config['score_labels'],
            num_layer=self.config['lstm_num_layer'],
            bilstm_flag=self.config['bidirectional_flag'],
            cache_dir=self.config["cache_dir_path"]
        )

        electra_model.cuda()

        is_demo = True

        while (is_demo):
            total_input_ids, total_attention_mask, total_token_type_ids, total_senti_seq, total_score_seq, total_word_seq = [], [], [], [], [], []
            score_labels = None
            senti_labels = None
            datas = input().strip()
            if datas == "-1":
                break
            tokens = electra_tokenizer.tokenize(datas)
            tokens = ["[CLS]"] + tokens
            tokens = tokens[:self.config['max_length']-1]
            tokens.append("[SEP]")

            input_ids = [electra_tokenizer._convert_token_to_id(token) for token in tokens]
            assert len(input_ids) <= self.config['max_length']

            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)

            padding = [0] * (self.config['max_length'] - len(input_ids))

            total_word_seq.append(len(input_ids))

            input_ids += padding
            attention_mask += padding
            token_type_ids += padding

            total_input_ids.append(input_ids)
            total_attention_mask.append(attention_mask)
            total_token_type_ids.append(token_type_ids)

            total_senti_seq.append([i for i in range(self.config['senti_labels'])])
            total_score_seq.append([i for i in range(self.config['score_labels'])])

            total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
            total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
            total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)
            total_senti_seq = torch.tensor(total_senti_seq, dtype=torch.long)
            total_score_seq = torch.tensor(total_score_seq, dtype=torch.long)
            total_word_seq = torch.tensor(total_word_seq, dtype=torch.long)


            dataset = TensorDataset(total_input_ids, total_attention_mask, total_token_type_ids, total_senti_seq,
                                    total_score_seq, total_word_seq)

            test_sampler = SequentialSampler(dataset)
            test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)

            electra_model.eval()


            for step, batch in enumerate(test_dataloader):
                batch = tuple(t.cuda() for t in batch)
                input_ids, attention_mask, token_type_ids, senti_seq, score_seq, word_len_seq = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

                # 입력 데이터에 대한 출력 결과 생성
                score_logits, senti_logits = electra_model(input_ids, attention_mask, token_type_ids,
                                                           None, score_seq, senti_seq, word_len_seq)

                senti_logits = senti_logits.squeeze()
                predict = F.softmax(senti_logits, dim=0)
                predict = predict.argmax(dim=-1)
                predict = predict.cpu().detach().numpy().tolist()
                input_ids = input_ids.cpu().detach().numpy().tolist()

                score_pred = F.softmax(score_logits, dim=1)
                score_pred = score_pred.argmax(dim=1)
                labels = score_pred.cpu().detach().numpy().tolist()

                tokens = [electra_tokenizer._convert_id_to_token(input_id) for input_id in input_ids[0]]
                sep_idx = tokens.index("[SEP]")
                # [CLS] 토큰 제거
                tokens = tokens[1:]

                # [SEP] 토큰 제거
                labels = labels[0][1:sep_idx]
                tokens = tokens[:tokens.index("[SEP]")]

                # 입력 sequence 복원
                # sequence = electra_tokenizer.convert_tokens_to_string(tokens)

                if (predict == 0):
                    predict = "부정"
                else:
                    predict = "긍정"

                print()
                for token, label in zip(tokens, labels):
                    if label == 2:
                        print(token.replace("##", "") + "{매우 부정}", end=" ")
                    elif label == 3:
                        print(token.replace("##", "") + "{부정}", end=" ")
                    elif label == 5:
                        print(token.replace("##", "") + "{긍정}", end=" ")
                    elif label == 6:
                        print(token.replace("##", "") + "{매우 긍정}", end=" ")
                    else:
                        print(token.replace("##", ""), end=" ")
                print("\n")
                print("감성 분석 결과 : {}".format(predict))
                print()



