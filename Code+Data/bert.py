from transformers import BertConfig,BertTokenizer,BertModel
import torch
import pickle
import pandas as pd

model_name = './FinBERT_L-12_H-768_A-12_pytorch'
tokenizer = BertTokenizer.from_pretrained(model_name)		# 通过词典导入分词器
model_config = BertConfig.from_pretrained(model_name)		# 导入配置文件
model = BertModel.from_pretrained(model_name, config=model_config)

def bertmodel(string):
    input_id = tokenizer.encode_plus(string)

    input_ids = torch.tensor(input_id['input_ids'])
    token_type_ids = torch.tensor(input_id['token_type_ids'])
    attention_mask_ids=torch.tensor(input_id['attention_mask'])

    # 将模型转化为eval模式
    model.eval()
    # 将模型和数据转移到cuda, 若无cuda,可更换为cpu
    device = 'cuda:0'
    tokens_tensor = input_ids.to(device).unsqueeze(0)
    segments_tensors = token_type_ids.to(device).unsqueeze(0)
    attention_mask_ids_tensors = attention_mask_ids.to(device).unsqueeze(0)
    model.to(device)
    with torch.no_grad():
        # See the models docstrings for the detail of the inputs
        outputs = model(tokens_tensor, segments_tensors, attention_mask_ids_tensors)
        # Transformers models always output tuples.
        # See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the Bert model
        encoded_layers = outputs
    return encoded_layers[1]


def concat_func(x):
    return pd.Series({
        '新闻标题':','.join(x['新闻标题'].unique())
    }
    )


data = pd.read_excel('./财经新闻/News_NewsInfo2022.xlsx')
data = data[['公布日期','新闻标题']]
data=data.groupby(data['公布日期']).head(10).reset_index()
data=data.groupby(data['公布日期']).apply(concat_func).reset_index()
data_new = pd.DataFrame(columns=['公布日期']+[i for i in range(1,769)])
for i in range(data.shape[0]):
    data_new.at[i, '公布日期'] = data.at[i, '公布日期']
    a = bertmodel(data.at[i, '新闻标题']).to('cpu').detach().numpy()
    list1 = a[0].tolist()
    for j in range(1,769):
        data_new.at[i,j]= list1[j-1]
data_new.to_csv('textfeature2022.csv',index=0)


