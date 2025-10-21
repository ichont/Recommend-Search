from sentence_transformers import SentenceTransformer

# 这个代码只需要成功运行一次
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./all-MiniLM-L6-v2/')  # 将模型保存到当前目录下的 local_model 文件夹
print("模型已保存至本地目录。")