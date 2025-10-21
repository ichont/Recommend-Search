# 检查依据模糊搜索的使用方法
- train.py训练模型和向量
- test.py进行推理测试

## 模型选择
- 文件夹safe_resources使用模型——paraphrase-multilingual-MiniLM-L12-v2,但参数量比较大，首选
- 文件夹safe_resources2使用模型all-MiniLM-L6-v2，比较适中，但是精度准确率不如上面那个

## 文件夹
模型命名的文件夹是原始模型，从huggingface下载的。

## 进展说明
模型还是用paraphrase-multilingual-MiniLM-L12-v2好点，500M总共
然后发布的话，API-server.py只能发布在本地回环或者本地局域网，暂时无法暴露在公网，暴露在公网建议部署在云服务器上
建议将这一套部署在云服务器以应对大规模访问和稳定性
