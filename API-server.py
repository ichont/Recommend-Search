from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import numpy as np

# 初始化Flask应用
app = Flask(__name__)
api = Api(app)

# 全局变量存储加载的资源
model = None
index = None
records = None


def load_resources(save_dir="./safe_resources"):
    """加载模型、索引和记录数据[1](@ref)"""
    global model, index, records

    try:
        # 加载句子嵌入模型
        model_path = os.path.join(save_dir, "sentence_model")
        model = SentenceTransformer(model_path)

        # 加载FAISS索引
        index_path = os.path.join(save_dir, "safety_index.faiss")
        index = faiss.read_index(index_path)

        # 加载记录数据
        data_path = os.path.join(save_dir, "texts_records.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        records = data["records"]

        print(f"✅ 资源加载完成，共加载 {len(records)} 条安全记录")
        return True

    except Exception as e:
        print(f"❌ 资源加载失败: {e}")
        return False


class SafetyQuery(Resource):
    """安全知识查询API资源类[6,7](@ref)"""

    def post(self):
        """处理安全知识查询请求"""
        # 检查资源是否已加载
        if model is None or index is None or records is None:
            return {
                "status": "error",
                "message": "系统资源未正确加载，请检查服务状态"
            }, 500

        # 获取请求数据
        data = request.get_json()

        if not data or 'query' not in data:
            return {
                "status": "error",
                "message": "缺少查询参数 'query'"
            }, 400

        query_text = data['query']
        top_k = data.get('top_k', 5)  # 默认返回5条结果

        try:
            # 执行查询
            results = self._execute_query(query_text, top_k)

            return {
                "status": "success",
                "query": query_text,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"查询处理失败: {str(e)}"
            }, 500

    def _execute_query(self, query_text, top_k=5):
        """执行安全知识查询[1](@ref)"""
        # 生成查询向量
        query_embedding = model.encode([query_text])

        # FAISS搜索
        distances, indices = index.search(query_embedding, top_k)

        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(records):
                result = records[idx].copy()
                result['similarity_score'] = float(1 / (1 + distances[0][i]))  # 转换为相似度分数
                result['rank'] = i + 1
                results.append(result)

        return results


class HealthCheck(Resource):
    """健康检查端点[8](@ref)"""

    def get(self):
        return {
            "status": "healthy" if model and index and records else "unhealthy",
            "records_loaded": len(records) if records else 0,
            "service": "safety-knowledge-api"
        }


# 注册API路由
api.add_resource(HealthCheck, '/health')
api.add_resource(SafetyQuery, '/api/query')


@app.route('/')
def index():
    """API主页"""
    return {
        "message": "安全知识查询API服务",
        "version": "1.0.0",
        "endpoints": {
            "健康检查": "/health (GET)",
            "安全查询": "/api/query (POST)"
        }
    }


if __name__ == '__main__':
    # 启动时加载资源
    print("🚀 启动安全知识查询API服务...")

    if load_resources():
        # 获取端口配置，默认5000
        port = int(os.environ.get('PORT', 5000))

        # 启动Flask应用
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ 服务启动失败：资源加载异常")