import torch
import torch.nn.functional as F
import argparse
from transformers import BertTokenizer,GPT2Config
from MyModel import MYGPT2LMHeadModel
import copy


def top_k_top_p(logits, top_k, top_p, filter_value=-float("Inf")):
    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))
    if top_k > 0:
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][-1, None]
            logit[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


def predict_one_sample(model, tokenizer, device, args, keyword):
    keyword_tokens = tokenizer.tokenize(keyword.replace(" ", "[SPACE]"))
    if len(keyword_tokens) > args.max_len - 3 - args.generate_max_len:
        keyword_tokens = keyword_tokens[:args.max_len - 3 - args.generate_max_len]
    unk_id = tokenizer.convert_tokens_to_ids("[UNK]")
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    if args.mode == "CCPC":
        content_id = tokenizer.convert_tokens_to_ids("[CONTENT]")
        keyword_id = tokenizer.convert_tokens_to_ids("[KEYWORD]")
        keyword_tokens = ["[CLS]"] + keyword_tokens+ ["[SEP]"]
        token_type_ids = [[keyword_id] * len(keyword_tokens) for _ in range(args.batch_size)]
        token_type_tensors = torch.tensor(token_type_ids).long().to(device)
        next_token_type = torch.tensor([[content_id] for _ in range(args.batch_size)]).long().to(device)
    else:
        keyword_tokens = ["[CLS]"] + keyword_tokens
    input_ids = tokenizer.convert_tokens_to_ids(keyword_tokens)
    input_ids = [copy.deepcopy(input_ids) for _ in range(args.batch_size)]
    input_tensors = torch.tensor(input_ids).long().to(device)
    generated = []
    keywords = []
    for i in input_ids[0]:
        keywords.append([i]*args.batch_size)
    finish_set = set()
    with torch.no_grad():
        for _ in range(args.generate_max_len):
            if args.mode == "CCPC":
                outputs = model.forward2(input_ids=input_tensors, token_type_ids=token_type_tensors)
            else:
                outputs = model(input_ids=input_tensors)
            next_token_logits = outputs.logits[:, -1, :]
            for index in range(args.batch_size):
                for token_id in set([token_ids[index] for token_ids in generated]):
                    next_token_logits[index][token_id] /= args.repetition_penalty
            for index in range(args.batch_size):
                for token_id in set([token_ids[index] for token_ids in keywords]):
                    next_token_logits[index][token_id] /= args.keyword_penalty
            for next_token_logit in next_token_logits:
                next_token_logit[unk_id] = -float("Inf")
            filter_logits = top_k_top_p(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            next_tokens = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)
            for index, token_id in enumerate(next_tokens[:, 0]):
                if token_id == sep_id:
                    finish_set.add(index)
            finish_flag = True
            for index in range(args.batch_size):
                if index not in finish_set:
                    finish_flag = False
                    break
            if finish_flag:
                break
            generated.append([token.item() for token in next_tokens[:, 0]])
            input_tensors = torch.cat((input_tensors, next_tokens), dim=-1)
            if args.mode =="CCPC":
                token_type_tensors = torch.cat((token_type_tensors, next_token_type), dim=-1)
        candidate_responses = []
        for index in range(args.batch_size):
            responses = []
            for token_index in range(len(generated)):
                if generated[token_index][index] != sep_id:
                    responses.append(generated[token_index][index])
                else:
                    break
            candidate_responses.append(
                "".join(tokenizer.convert_ids_to_tokens(responses)).replace("##", "").replace("[SPACE]",""))
    return candidate_responses

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=True, type=bool, help='是否使用GPU')
    parser.add_argument('--mode', default="CCPC", type=str, help='训练模式(唐诗:tang、宋诗：song、宋词：ci、关键词诗：CCPC')
    parser.add_argument('--batch_size', default=5, type=int, help='生成古诗的个数')
    parser.add_argument('--generate_max_len', default=120, type=int, help='生成古诗的最大长度')
    parser.add_argument('--repetition_penalty', default=1.4, type=float, help='重复处罚率')
    parser.add_argument('--keyword_penalty', default=1.2, type=float, help='关键词处罚率')
    parser.add_argument('--top_k', default=15, type=float, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=0.95, type=float, help='解码时保留概率累加大于多少的标记')

    return parser.parse_args()


args = set_args()

device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
tokenizer = BertTokenizer(vocab_file="./model_"+args.mode+"/vocab.txt",do_lower_case=False)
if args.mode == "CCPC":
    tokenizer.add_tokens("[SPACE]", special_tokens=True)
    tokenizer.add_tokens("[KEYWORD]", special_tokens=True)
    tokenizer.add_tokens("[CONTENT]", special_tokens=True)
model = MYGPT2LMHeadModel.from_pretrained("./model_"+args.mode+"/final_model")
args.max_len = model.config.n_ctx
model.to(device)
model.eval()

print('开始生成古诗，输入CTRL + Z退出')

while True:
    content = input("输入的字符为:")
    args.generate_max_len = min(args.generate_max_len, args.max_len - 5 - len(content))
    poems = predict_one_sample(model, tokenizer, device, args, content)
    for i, poem in enumerate(poems):
        if args.mode !="CCPC":
            print("生成的第{}个诗为：{}".format(i + 1, content+poem))
        else:
            print("生成的第{}个诗为：{}".format(i + 1, poem))
