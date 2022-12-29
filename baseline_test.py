from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json,pylcs
from rouge import Rouge
import numpy as np
import argparse
import os
from softEmbedding import SoftEmbedding
from model import PromptModel
task_dict={
    'mrc': 0,
    'classify': 1,
    'anaphora_resolution': 1,
    'nli': 2,
    'generate': 3
}

def preprocess(text):
  return text.replace("\n", "_")
def postprocess(text):
  return text.replace("_", "\n")

def answer_fn(text, type, sample=False, top_p=0.6):
  '''sample：是否抽样。生成任务，可以设置为True;
     top_p：0-1之间，生成的内容越多样、
  '''
  text = preprocess(text)
  encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt")
  if args.use_prompt:
    encoding['input_ids'] = torch.cat([torch.full((1,args.max_prompt_length), 0), encoding['input_ids']], dim=1)
    encoding['attention_mask'] = torch.cat([torch.full((1,args.max_prompt_length), 1), encoding['attention_mask']], dim=1)
    encoding['types'] = torch.tensor([type], dtype=torch.long)
  encoding = encoding.to(device)
  # types参数会通过generate的**model_kwargs自动传到model的forward中
  if not sample: # 不进行采样
    out = model_trained.generate(**encoding, return_dict_in_generate=True, output_scores=True, max_length=128, num_beams=4, length_penalty=0.6)
  else: # 采样（生成）
    out = model_trained.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128, do_sample=True, top_p=top_p)

  
  out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
  return postprocess(out_text[0]), out['sequences_scores'] # text, log_scores


# 在公开测试集上做预测，并写入到文件
def predict_on_test(source_file, target_file, select_top=-1):
  lines=open(source_file,'r').readlines()
  if select_top!=-1: # select_top==-1 -->全量预测；其他值，则选取top值进行预测
    lines=lines[0:select_top] 
  print("length of lines:",len(lines))
  target_object=open(target_file,'w')
  for i,line in enumerate(lines):
    # print(i,line)
    json_string_right=json.loads(line)
    input_string=json_string_right["input"]
    target_answer=json_string_right["target"]
    type=task_dict[json_string_right["type"]]

    predict_answer, log_score=answer_fn(input_string, type)
    score = torch.exp(log_score)
    json_string_predict={"target":predict_answer.strip(),"type":json_string_right["type"], 'score':float(score)}
    json_string_predict=json.dumps(json_string_predict,ensure_ascii=False)
    target_object.write(json_string_predict+"\n")
    if i%100==0: 
      print(i,"input_string:",input_string,";predict:",predict_answer)

# 使用评估脚本进行评估
"""
脚本见：https://github.com/CLUEbenchmark/pCLUE/blob/main/evaluate_pclue.py
计算pCLUE任务总分，及子分数
"""
def f1_sim(text_a, text_b):
    """F1相似度
    说明：算出两个文本的最长公共子序列长度，然后乘2并处以两者
    长度之和。推荐用pylcs算，速度较快。
    """
    if not text_a and not text_b:
        return 0.
    else:
        lcs = pylcs.lcs(text_a, text_b)
        return 2. * lcs / (len(text_a) + len(text_b))

def rouge_l_zh(target, pred):
    """计算Rouge-l得分，Rouge-l指标常用于评估自动文本摘要及翻译任务
    target: 真实标签
    pred: 预测标签"""
    if not(isinstance(target, str) or isinstance(pred, str)):
        logger.info("target或pred为非字符串！请检查!")
        return
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(list(pred)), " ".join(list(target)))
        score = scores[0]["rouge-l"]
        return score["f"]

def normalize(text):
    """简单的文本标准化
    """
    return ' '.join(text.lower().split())

def evaluate_pclue_fn(predict_file, target_file, select_top=-1):
    """
    计算pclue的成绩
    :param predict_file: 预测文件
    :param target_file:  正确的文件
    :return: 一个dict，包括总分score，以及各个部分的分数（mrc, generate, classify, nli）
    """
    predict_lines=open(predict_file,'r').readlines()
    target_lines=open(target_file,'r').readlines()
    
    if select_top!=-1:
        predict_lines=predict_lines[0:select_top]
        target_lines=target_lines[0:select_top]

    # 1.记录
    classify_list=[]
    mrc_list=[]
    generate_list=[]
    nli_list=[]
    for i, target_line in enumerate(target_lines):
        # e.g. target_line = {"target": "不同"}
        predict_line=predict_lines[i]
        target_answer=json.loads(target_line.replace("，",","))["target"] # 正确的标签
        if isinstance(target_answer, list):  # 将列表转换为字符串，如关键词生成
            target_answer = "，".join(target_answer)
        target_answer=normalize(target_answer)
        predict_answer=json.loads(predict_line)["target"] # 预测的标签
        predict_answer=normalize(predict_answer)
        if len(predict_answer)==0: 
          predict_answer="无答案"
        if i%100==0:
          print(i,"target_answer:",target_answer,";predict_answer:",predict_answer,"length of predict_answer:",len(predict_answer))

        type=json.loads(target_line.replace("，",","))["type"] # 替换可能存在问题的数据，如有，以便能加载为json
        if type=='classify' or type=='anaphora_resolution': # 分类
            label_temp=True if target_answer==predict_answer else False
            classify_list.append(label_temp)
        elif type=='mrc': # 阅读理解
            em=1 if target_answer==predict_answer else 0
            f1=f1_sim(predict_answer,target_answer)
            mrc_list.append((em, f1))
        elif type=='generate': # 生成
            rouge_l=rouge_l_zh(target_answer, predict_answer)
            generate_list.append(rouge_l)
        elif type=='nli': # 推理
            label_temp = True if target_answer == predict_answer else False
            nli_list.append(label_temp)
        else:
            print("error...predict_line:",predict_line,";target_line:",target_line)
            break # 中断运行
        # if predict_answer==target_answer: count_right=count_right+1
        if i<10: print(i, 'target_answer:',target_answer,";predict_answer:",predict_answer) # 显示部分内容

    # 2.计算最后的得分
    classify_score=np.average(classify_list)
    nli_score=np.average(nli_list)
    generate_score=np.average(generate_list)
    mrc_em_score=np.average([x[0] for x in mrc_list])
    mrc_f1_score=np.average([x[1] for x in mrc_list])
    mrc_score=np.average([mrc_em_score,mrc_f1_score])
    # 计算总分
    score=np.average([classify_score,nli_score,generate_score,mrc_score])
    # 保存分数
    result_dict={"score":score,"classify_score":classify_score,"nli_score":nli_score,"generate_score":generate_score,
                 "mrc_em_score":mrc_em_score,"mrc_f1_score":mrc_f1_score}
    return result_dict

if __name__=='__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',type=str,default='outputs/general_test')
    parser.add_argument('--data_dir',type=str,default='pclue_data')
    parser.add_argument('--test_file', type=str, default='pCLUE_test_public.json')
    parser.add_argument('--tokenizer_path', type=str, default='PLM/mengziT5MT')
    parser.add_argument('--model_path', type=str, default='outputs/general_test')
    parser.add_argument('--max_prompt_length', type=int, default=10)
    parser.add_argument('--select_top', type=int, default=-1) # 数据条数，-1表示全量
    parser.add_argument('--do_metrics', action='store_true') # default: False
    parser.add_argument('--use_prompt', action='store_true') # default: False
    parser.add_argument('--prompt_type', type=str, default='from_vocab')
    parser.add_argument('--prompt_paradigm', type=str, default='single')
    parser.add_argument('--num_tasks', type=int, default=4)
    args, _ = parser.parse_known_args()
    print(args)
    assert args.prompt_type in ['from_vocab', 'random'], 'prompt type error'
    assert args.prompt_paradigm in ['single', 'multi'], 'prompt type error'

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path) #ClueAI/PromptCLUE ,注意这里是下面用的模型的对应的
    if args.use_prompt:
      initialize_from_vocab = True if args.prompt_type == 'from_vocab' else False

      model_trained = PromptModel(args.model_path, args.max_prompt_length, num_tasks=args.num_tasks, \
       initialize_from_vocab=initialize_from_vocab, prompt_paradigm=args.prompt_paradigm).model

      init_state_dict = torch.load(os.path.join(args.model_path, 'pytorch_model.bin'), map_location="cpu")
      model_trained.load_state_dict(init_state_dict)
      print(model_trained)
      # 测试加载前后是否相等
      count = 0
      for key in model_trained.state_dict().keys():
        # (init_state_dict[key] != model_trained.state_dict()[key]).any()
        if not init_state_dict[key].equal(model_trained.state_dict()[key]):
          count+=1
      assert count==0, "difference between train and test"
    else:
      model_trained = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    model_trained.to(device)

    # 得到预测结果
    select_top = args.select_top # TODO 改变select_top的值，使得用一个大的数量，或全量数据
    source_file = os.path.join(args.data_dir, args.test_file)
    test_file_prefix = os.path.splitext(args.test_file)[0]
    target_file = os.path.join(args.output_dir, test_file_prefix+'_predict.json')
    predict_on_test(source_file, target_file, select_top)

    # 评测
    if args.do_metrics:
      result=evaluate_pclue_fn(target_file, source_file, select_top)
      print("result:",result)
