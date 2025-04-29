import csv
import os
from langchain_ollama import OllamaLLM
import re

def create_questionnaire(topic, num_questions=4):
    """使用AI生成問卷問題"""
    try:
        # 初始化Ollama模型
        llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.7)
        
        # 生成問卷的提示 (確保使用繁體中文)
        prompt = f"""請使用繁體中文，針對「{topic}」主題，設計{num_questions}個問卷問題。
        問題應該包含下列各種類型，並明確標示格式要求：
        - 基本資料問題（例如：「請問您的職業是？」）
        - 評分類問題（必須標示為 (1-5分)，例如：「請評價此產品的使用體驗 (1-5分)」）
        - 選擇題（必須列出選項，例如：「您最常使用本服務的時間？(上午/下午/晚上)」）
        - 開放式意見問題（例如：「您對於{topic}有什麼建議？」）
        
        只需回傳問題清單，每行一個問題，確保使用繁體中文，並明確標示評分和選項格式。
        不需要編號或其他說明。"""
        
        # 獲取AI回應並處理成列表
        response = llm.invoke(prompt)
        
        # 清理回應，移除可能的思考標記和其他額外內容
        cleaned_response = clean_ai_response(response)
        
        # 分割為獨立問題
        questions = [q.strip() for q in cleaned_response.strip().split('\n') if q.strip()]
        
        # 移除問題前的編號和標籤，但保留格式提示如(1-5分)
        questions = [remove_question_markers(q) for q in questions]
        
        # 確保問題符合格式要求
        questions = [ensure_question_format(q, topic) for q in questions]
        
        # 確保我們有足夠的問題
        while len(questions) < num_questions:
            if len(questions) % 4 == 0:
                questions.append(f"請問您的年齡範圍？(20歲以下/20-30歲/31-40歲/41歲以上)")
            elif len(questions) % 4 == 1:
                questions.append(f"您對{topic}的滿意度為？ (1-5分)")
            elif len(questions) % 4 == 2:
                questions.append(f"您使用{topic}的頻率為？(每天/每週/每月/極少)")
            else:
                questions.append(f"您對於{topic}還有什麼其他建議或意見？")
            
        return questions[:num_questions]  # 確保只返回要求的問題數量
    except Exception as e:
        print(f"無法連接AI模型生成問卷: {e}")
        print("使用預設問題...")
        # 預設問題，以防AI生成失敗 (確保使用繁體中文)
        return [
            f"請問您對{topic}的了解程度？ (1-5分)",
            f"您認為{topic}最重要的特點是什麼？",
            f"您使用{topic}相關服務的頻率為？ (從不/偶爾/經常/總是)",
            f"您對{topic}有什麼建議或意見？"
        ][:num_questions]  # 確保只返回要求的問題數量

def clean_ai_response(response):
    """清理AI回應中的思考過程和其他不必要內容"""
    # 移除<think>...</think>標記及其內容
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # 移除可能的JSON或其他格式標記
    response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
    
    return response

def remove_question_markers(question):
    """移除問題前的編號和前導標籤，但保留格式提示如(1-5分)"""
    # 移除前面的編號，如 "1. ", "Q1: "等
    question = re.sub(r'^(\d+\.\s*|\w+\d+:\s*)', '', question)
    
    # 移除可能的問題類型標籤，但保留括號內的評分或選項提示
    question = re.sub(r'^\s*\[.*?\]\s*', '', question)
    
    return question.strip()

def ensure_question_format(question, topic):
    """確保問題符合格式要求，包含評分標示或選項"""
    # 檢查是否已包含評分標示
    has_rating = re.search(r'\(\s*\d+\s*[-–—]\s*\d+\s*分\s*\)', question) is not None
    
    # 檢查是否已包含選項
    has_options = re.search(r'\([^)]+\/[^)]+\)', question) is not None
    
    # 如果是評分相關問題但沒有評分標示，添加(1-5分)
    if (re.search(r'評分|評價|滿意度|分數|評估', question) and not has_rating 
            and not '評價此產品' in question):
        # 找到問號的位置
        q_pos = question.find('？')
        if q_pos == -1:
            q_pos = len(question)
        
        # 在問號前插入評分標示
        question = question[:q_pos] + ' (1-5分)' + question[q_pos:]
    
    # 如果是頻率相關問題但沒有選項，添加頻率選項
    if (re.search(r'頻率|多久|多常|頻繁|經常', question) and not has_options
            and not '頻率為' in question):
        # 找到問號的位置
        q_pos = question.find('？')
        if q_pos == -1:
            q_pos = len(question)
        
        # 在問號前插入選項
        question = question[:q_pos] + ' (從不/偶爾/經常/總是)' + question[q_pos:]
    
    # 將問題中可能的簡體字轉為繁體字 (這裡列出常見簡繁不同的詞)
    simplify_to_traditional = {
        '设计': '設計', '问卷': '問卷', '调查': '調查', '专业': '專業',
        '满意度': '滿意度', '评价': '評價', '购买': '購買', '选择': '選擇',
        '频率': '頻率', '经常': '經常', '产品': '產品', '质量': '質量',
        '价格': '價格', '服务': '服務', '体验': '體驗', '意见': '意見',
        '问题': '問題', '建议': '建議', '改进': '改進', '优点': '優點', 
        '缺点': '缺點', '使用': '使用', '购物': '購物', '消费': '消費'
    }
    
    for simplified, traditional in simplify_to_traditional.items():
        question = question.replace(simplified, traditional)
    
    return question

def collect_responses(questions):
    """收集用戶對問卷的回答"""
    responses = []
    print("\n===== 開始問卷調查 =====")
    for i, q in enumerate(questions, 1):
        print(f"問題 {i}: {q}")
        user_ans = input("您的回答: ")
        responses.append(user_ans)
    print("===== 問卷完成 =====\n")
    return responses

def save_results(questions, responses, file_path=None):
    """儲存問卷結果到指定位置"""
    if not file_path:
        file_path = input("請輸入儲存問卷結果的檔案路徑 (預設為'問卷調查結果.csv'): ")
        if not file_path:
            file_path = "問卷調查結果.csv"
    
    # 確保檔案路徑有效
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # 確保問題和回答的列表長度一致
    if len(questions) != len(responses):
        print("警告：問題數量與回答數量不匹配。")
        min_length = min(len(questions), len(responses))
        questions = questions[:min_length]
        responses = responses[:min_length]
    
    try:
        # 檢查檔案是否存在
        file_exists = os.path.isfile(file_path)
        
        if file_exists:
            # 如果檔案存在，讀取現有內容
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_data = list(reader)
                
            # 檢查現有標題與新問題是否匹配
            if existing_data and len(existing_data[0]) == len(questions):
                # 如果問題數量匹配，添加新行
                with open(file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(responses)
            else:
                # 問題數量不匹配，創建新檔案
                print("警告：問題與現有檔案不匹配，將創建備份並生成新檔案。")
                # 備份原檔案
                backup_path = file_path + ".backup"
                with open(backup_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(existing_data)
                print(f"原檔案已備份到 {backup_path}")
                
                # 創建新檔案
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(questions)
                    writer.writerow(responses)
        else:
            # 如果檔案不存在，創建新檔案
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(questions)
                writer.writerow(responses)
        
        print(f"問卷結果已儲存至 {file_path}！")
        return file_path
    except Exception as e:
        print(f"儲存檔案時發生錯誤: {e}")
        # 嘗試備份儲存
        backup_path = file_path + ".backup"
        try:
            with open(backup_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(questions)
                writer.writerow(responses)
            print(f"問卷結果已備份儲存至 {backup_path}")
            return backup_path
        except:
            print("無法儲存問卷結果")
            return None

def preview_csv(file_path):
    """預覽CSV檔案的格式"""
    if not os.path.exists(file_path):
        print(f"檔案 {file_path} 不存在")
        return
    
    print(f"\n===== CSV檔案預覽 ({file_path}) =====")
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    print("標題列: " + " | ".join(row))
                else:
                    print(f"資料列 {i}: " + " | ".join(row))
                if i >= 3:  # 只預覽前幾行
                    print("...")
                    break
        print("===== 預覽結束 =====\n")
    except Exception as e:
        print(f"預覽檔案時發生錯誤: {e}")

def main():
    """主程式流程"""
    print("==== AI問卷調查系統 ====")
    
    # 取得用戶想要的問卷主題
    topic = input("請輸入您想要調查的主題: ")
    if not topic:
        topic = "產品滿意度"
        print(f"未輸入主題，使用預設主題: {topic}")
    
    # 詢問所需問題數量
    try:
        num_q = int(input("請問需要幾個問題 (預設為4): ") or "4")
    except ValueError:
        num_q = 4
        print("輸入無效，使用預設問題數量: 4")
    
    # 生成問卷
    print(f"\n正在生成有關「{topic}」的問卷問題...")
    questions = create_questionnaire(topic, num_q)
    
    print("\n問卷已生成，包含以下問題:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    # 詢問是否要開始收集回答
    start = input("\n是否開始收集回答？(y/n): ").lower()
    if start != 'y':
        print("已取消問卷調查")
        return
    
    # 收集回答
    responses = collect_responses(questions)
    
    # 儲存結果
    output_file = input("請輸入儲存檔案路徑 (直接按Enter使用預設路徑): ")
    file_path = save_results(questions, responses, output_file)
    
    # 預覽CSV檔案
    if file_path:
        preview_csv(file_path)

if __name__ == "__main__":
    main()
