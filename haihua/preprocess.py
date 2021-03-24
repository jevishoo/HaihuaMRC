import json
import pandas as pd

with open('./data/train.json','r',encoding='utf-8')as f: #读入json文件
    train_data = json.load(f)

train_df = []

for i in range(len(train_data)): #将每个文章-问题-答案作为一条数据
    data = train_data[i]
    content = data['Content']
    questions = data['Questions']
    for question in questions:
        question['Content'] = content
        train_df.append(question)

train_df =  pd.DataFrame(train_df) #转换成csv表格更好看一点


with open('./data/validation.json','r',encoding='utf-8')as f:
    test_data = json.load(f)

test_df = []

for i in range(len(test_data)):
    data = test_data[i]
    content = data['Content']
    questions = data['Questions']
    cls = data['Type']
    diff = data['Diff']
    for question in questions:
        question['Content'] = content
        question['Type'] = cls
        question['Diff'] = diff
        test_df.append(question)

test_df =  pd.DataFrame(test_df)

train_df['label'] = train_df['Answer'].apply(lambda x:['A','B','C','D'].index(x)) #将标签从ABCD转成0123
test_df['label'] = 0

train_df.to_csv('./data/train.csv',index=False)
test_df.to_csv('./data/test.csv',index=False)
