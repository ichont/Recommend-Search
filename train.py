# save_model.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import pickle
import re


def parse_data(file_path):
    """
    解析文本数据文件，返回记录列表
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 按记录分割
    record_blocks = content.split('#\n')

    for block in record_blocks:
        if not block.strip():
            continue

        record = {}
        lines = block.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                record[key.strip()] = value.strip()

        if record:  # 确保非空记录
            records.append(record)

    return records


def init_embedding_model():
    """
    初始化句子嵌入模型
    """
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2") # 切换你的模型


def generate_search_text(record):
    """
    从记录中生成用于搜索的文本
    """
    keys = ['隐患描述', '检查依据', '整改建议', '检查对象']
    return ' '.join(str(record.get(k, '')) for k in keys)


def create_vector_index(records, model):
    """
    创建FAISS向量索引
    """
    texts = [generate_search_text(r) for r in records]
    embeddings = model.encode(texts, show_progress_bar=True)

    # 创建FAISS索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    return index, texts


def save_resources(model, index, texts, records, save_dir="./safe_resources"): # 修改1
    """
    保存所有资源到本地目录 [1,2](@ref)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型（使用SentenceTransformer自带的保存方法）
    model.save(os.path.join(save_dir, "sentence_model"))

    # 保存FAISS索引
    faiss.write_index(index, os.path.join(save_dir, "safety_index.faiss"))

    # 保存文本和记录（使用pickle）[4,5](@ref)
    with open(os.path.join(save_dir, "texts_records.pkl"), "wb") as f:
        pickle.dump({"texts": texts, "records": records}, f)

    # 保存模型信息
    model_info = {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2", # 修改4
        "dimension": index.d,
        "num_records": len(records)
    }

    with open(os.path.join(save_dir, "model_info.pkl"), "wb") as f:
        pickle.dump(model_info, f)

    print(f"✅ 资源已保存到 {save_dir} 目录")
    print(f"📊 保存了 {len(records)} 条安全记录")
    print(f"🔢 向量维度: {index.d}")


def main():
    """
    主函数：加载数据并保存所有资源
    """
    FILE_PATH = 'data.txt'  # 修改为实际文件路径
    SAVE_DIR = "./safe_resources" # 修改3，换模型要改4个地方

    try:
        # 初始化模型
        print("🔄 初始化嵌入模型...")
        model = init_embedding_model()

        # 解析数据
        print("📖 解析数据文件...")
        records = parse_data(FILE_PATH)
        print(f"✅ 成功加载 {len(records)} 条安全记录")

        # 创建向量索引
        print("🔧 创建向量索引...")
        index, texts = create_vector_index(records, model)

        # 保存所有资源
        print("💾 保存资源到本地...")
        save_resources(model, index, texts, records, SAVE_DIR)

        print("🎉 模型保存完成！现在您可以运行 test.py 进行快速查询")

    except FileNotFoundError:
        print(f"❌ 错误：找不到数据文件 {FILE_PATH}")
        print("请确保 data.txt 文件存在于当前目录")
    except Exception as e:
        print(f"❌ 发生错误：{e}")


if __name__ == "__main__":
    main()