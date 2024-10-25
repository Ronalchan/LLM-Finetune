import requests
import json


# 构建 API 请求函数
def call_api(prompt):
    url = "http://localhost:8000/v1/chat/completions"
    
    # 构建请求头
    headers = {
        'Content-Type': 'application/json',
    }
    
    # 构建请求体
    data = {
        "model": "string",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "do_sample": True,
        "temperature": 0,
        "top_p": 0,
        "n": 1,
        "max_tokens": 0,
        "stop": "string",
        "stream": False
    }
    
    # 发送 POST 请求
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # 解析响应
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return f"Request failed with status code {response.status_code}: {response.text}"

def process_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 遍历每条记录，并对 input 执行 API 调用
    for entry in data:
        prompt = entry.get('instruction', '') + "\n" + entry.get('input', '')
        prediction = call_api(prompt)
        entry['predict'] = prediction  # 添加新的 predict 字段
        print(prediction)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

name = "train_4_classes.json"
output_name = name.replace('test', 'predict')
input_path = f"/scratch/chenqian/LLaMA-Factory/code_dataset/{name}"
output_path = f"/scratch/chenqian/LLaMA-Factory/code_dataset/eval/raw-{output_name}"

process_data(input_path, output_path)