# 检查依据模糊搜索的使用方法

一种利用向量空间的简易推荐搜索方法，以安全生产检查依据搜索为例

- train.py训练模型和向量
- test.py进行推理测试

- 先运行download.py下载huggingface的预训练模型
- 运行train.py保存模型和向量文件和索引至safe_resources文件夹中
- 运行test.py即可

## 模型选择
- 文件夹safe_resources使用模型——paraphrase-multilingual-MiniLM-L12-v2,但参数量比较大，首选
- 文件夹safe_resources2使用模型all-MiniLM-L6-v2，比较适中，但是精度准确率不如上面那个

## 文件夹
模型命名的文件夹是原始模型，从huggingface下载的。

## 版本需求
`numpy==2.3.3`
`faiss-cpu==1.12.0`
`sentence-transformers==5.1.1`
`Flask==3.1.2`
`Flask-RESTful==0.3.10`

## 进展说明
可以进行相关性搜索了，根据相关度输出前5条检查依据案例
模型还是用paraphrase-multilingual-MiniLM-L12-v2好点，500M总共
然后发布的话，API-server.py只能发布在本地回环或者本地局域网，暂时无法暴露在公网，暴露在公网建议部署在云服务器上
建议将这一套部署在云服务器以应对大规模访问和稳定性




