import json
from sklearn.metrics import classification_report

# 读取 JSON 文件
def load_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# 计算分类报告
def calculate_metrics(data):
    true_labels = []
    pred_labels = []
    label_names = set()  # 用于保存所有标签名
    
    for entry in data:
        true_label = entry["output"].split(" for ")[-1].strip()  # 从 output 中提取真实标签
        pred_label = entry["predict"].split(" for ")[-1].strip()  # 从 predict 中提取预测标签
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        # 添加到标签名集合
        label_names.add(true_label)
        label_names.add(pred_label)

    # 生成分类报告
    report = classification_report(true_labels, pred_labels, labels=list(label_names), zero_division=0)
    
    return report

# 主程序
def main():
    # filename = "predict_1_potential_solution_design.json"
    # filename = "predict_2_solution_review.json"
    # filename = "predict_3_code_implementation.json"
    filename = 'predict_4_classes.json'
    data = load_json_file(filename)
    report = calculate_metrics(data)
    
    print('Classification Report:')
    print(report)

if __name__ == "__main__":
    main()
