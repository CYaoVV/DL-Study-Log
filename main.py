import os
import pandas as pd
import torch
os.makedirs(os.path.join("..","data"),exist_ok=True)
data_file=os.path.join("..","data","house.csv")
with open(data_file,"w") as f:
    f.write("Num,Alley,Price\n")
    f.write("NA,Pave,127500\n")
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write('NA,NA,140000\n')
data=pd.read_csv(data_file)
print(data)
inputs=data.iloc[:,0:2]
outputs=data.iloc[:,2]
# inputs=inputs.fillna(inputs.mean(numeric_only=True))

inputs["Num"] = inputs["Num"].fillna(inputs["Num"].astype(float).mean(numeric_only=True))
# pd.read_csv 返回的是一个二维表格数据类型dataframe，比如SQL，Excel，csv等文件都有相应的语句可以转为这种数据结构方便后续运算
#float默认为float64 如需特别说明，写成astype("float32")  astype是as type的缩写，fillna从左到右依次执行
# 分别先访问Num列，再将这列转换为float型，然后再把所有的数值类型的求均值
inputs=pd.get_dummies(inputs,dummy_na=True)
print(inputs)
# 访问Dataframe类型的数据多列要打双重[]，如下，如果只引用一列，那么直接类似["Alley_Pave"]即可
inputs[["Alley_Pave","Alley_nan"]]=inputs[["Alley_Pave","Alley_nan"]].astype(int)
print(inputs)
# 回顾下Numpy张量和tensor的转换语法如下
# A=X.numpy() 这是张量X转为numpy张量A     B=torch.tensor(A) 转回tensor B
X,y=torch.tensor(inputs.values),torch.tensor(outputs.values)
# values 是 Pandas 的一个属性，返回一个 NumPy 数组，表示 DataFrame 中的所有值。
# 它本质上是将 Pandas 的 DataFrame 转换为 NumPy 格式，便于后续运算。
# 仅返回数据部分， 忽略DataFrame 的列名(第一行)和行索引（打印出来的第一列的那玩意）。
print(X,y)
