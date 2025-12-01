import random

def random_select_lines(input_file, output_file, n):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    if n > len(lines):
        raise ValueError(f"文件中的行数不足 {n} 行")

    selected_lines = random.sample(lines, n)  # 不放回抽样

    with open(output_file, 'w') as f:
        f.writelines(selected_lines)

    print(f"已成功从 {input_file} 中随机选取 {n} 行，保存至 {output_file}")

# 示例用法
if __name__ == '__main__':
    input_path = '/home/dww/OD/Work6/data/initial/coco_train_IRDST1.txt'       # 替换为你的输入文件路径
    output_path = 'random_IRDST1000.txt' # 输出文件名
    num_lines = 1000                       # 想要随机选多少行 100 200 500 1000

    random_select_lines(input_path, output_path, num_lines)