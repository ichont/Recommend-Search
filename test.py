# query_model.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os


def load_resources(save_dir="./safe_resources"): # 修改1，两个地方
    """
    从本地加载所有资源 [2,5](@ref)
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"资源目录不存在: {save_dir}")

    # 加载模型
    model_path = os.path.join(save_dir, "sentence_model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print("🔄 加载句子嵌入模型...")
    model = SentenceTransformer(model_path)

    # 加载FAISS索引
    index_path = os.path.join(save_dir, "safety_index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"索引文件不存在: {index_path}")

    print("🔧 加载FAISS索引...")
    index = faiss.read_index(index_path)

    # 加载文本和记录
    data_path = os.path.join(save_dir, "texts_records.pkl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    texts = data["texts"]
    records = data["records"]

    # 加载模型信息
    info_path = os.path.join(save_dir, "model_info.pkl")
    if os.path.exists(info_path):
        with open(info_path, "rb") as f:
            model_info = pickle.load(f)
        print(f"📊 加载了 {model_info['num_records']} 条安全记录")
        print(f"🔢 向量维度: {model_info['dimension']}")

    return model, index, texts, records


def safety_query(query_text, model, index, records, top_k=5):
    """
    执行安全知识查询
    """
    # 生成查询向量
    query_embedding = model.encode([query_text])

    # FAISS搜索
    distances, indices = index.search(query_embedding, top_k)

    # 获取相关记录
    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(records):  # 有效索引
            result = records[idx].copy()
            result['相似度得分'] = f"{1 / (1 + distances[0][i]):.3f}"  # 转换为相似度分数
            results.append(result)

    return results


def display_results(results, query_text):
    """
    美观地显示查询结果
    """
    print(f"\n🔍 查询: '{query_text}'")
    print("=" * 60)

    if not results:
        print("❌ 未找到相关记录")
        return

    print(f"✅ 找到 {len(results)} 条相关记录:\n")

    for i, res in enumerate(results, 1):
        print(f"📋 记录 {i} (相似度: {res.get('相似度得分', 'N/A')})")
        print("-" * 40)

        # 显示主要字段
        important_fields = ['隐患描述', '检查依据', '整改建议', '检查对象', '风险等级']
        for field in important_fields:
            if field in res and res[field]:
                print(f"  📌 {field}: {res[field]}")

        # 显示其他字段
        other_fields = [k for k in res.keys() if k not in important_fields + ['相似度得分']]
        if other_fields:
            for field in other_fields:
                if res[field]:  # 只显示非空字段
                    print(f"  📄 {field}: {res[field]}")

        print()  # 记录之间的空行


def main():
    """
    主函数：加载资源并提供交互式查询
    """
    SAVE_DIR = "./safe_resources" # 修改2

    try:
        # 检查资源是否存在
        if not os.path.exists(SAVE_DIR):
            print("❌ 资源目录不存在，请先生成资源文件")
            print("运行命令: python save_model.py")
            return

        # 加载资源
        print("🚀 正在加载安全知识库资源...")
        model, index, texts, records = load_resources(SAVE_DIR)

        print("🎉 资源加载完成！现在可以开始查询")
        print("💡 输入 'exit' 退出程序")
        print("💡 输入 'help' 查看帮助\n")

        while True:
            try:
                user_query = input("请输入安全检查查询: ").strip()

                if user_query.lower() == 'exit':
                    print("👋 再见！")
                    break

                if user_query.lower() == 'help':
                    print("\n📖 帮助信息:")
                    print("  - 输入具体的安全问题，如 '高空作业防护'")
                    print("  - 输入风险类型，如 '火灾风险'")
                    print("  - 输入设备名称，如 '起重机检查'")
                    print("  - 输入 'exit' 退出程序")
                    print("  - 输入 'help' 显示此帮助\n")
                    continue

                if not user_query:
                    continue

                # 执行查询
                results = safety_query(user_query, model, index, records)

                # 显示结果
                display_results(results, user_query)

            except KeyboardInterrupt:
                print("\n👋 用户中断，程序退出")
                break
            except Exception as e:
                print(f"❌ 查询过程中发生错误: {e}")
                continue

    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("请先运行 save_model.py 生成资源文件")
    except Exception as e:
        print(f"❌ 加载资源时发生错误: {e}")


if __name__ == "__main__":
    main()