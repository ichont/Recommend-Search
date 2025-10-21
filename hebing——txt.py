import os


def merge_txt_files(folder_path, output_file="dataall.txt"):
    """
    合并文件夹下的所有txt文件为一个文件
    :param folder_path: 存放txt文件的文件夹路径
    :param output_file: 合并后的输出文件名（默认: dataall.txt）
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在！")
        return

    # 获取所有txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    if not txt_files:
        print(f"警告：文件夹 '{folder_path}' 中没有找到txt文件！")
        return

    # 按文件名排序（可选）
    txt_files.sort()

    # 合并内容
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            file_path = os.path.join(folder_path, txt_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(f"===== 文件: {txt_file} =====\n")  # 可选：标记来源文件
                    outfile.write(infile.read())
                    outfile.write("\n\n")  # 文件间加空行分隔
                print(f"已合并: {txt_file}")
            except Exception as e:
                print(f"读取文件 '{txt_file}' 失败: {e}")

    print(f"\n合并完成！结果已保存到: {output_file}")


# 使用示例
folder_path = "./data"  # 替换为你的文件夹路径
merge_txt_files(folder_path)