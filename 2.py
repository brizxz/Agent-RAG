import logging
import os
import datetime
from langchain_ollama import OllamaLLM

# 配置日誌
def setup_logging(log_dir="logs"):
    """設置日誌系統"""
    # 確保日誌目錄存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日誌文件名（使用當前日期和時間）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"document_compare_{timestamp}.log")
    
    # 配置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同時輸出到控制台
        ]
    )
    
    logging.info(f"日誌初始化完成，日誌文件：{log_file}")
    return logging.getLogger()

# 初始化日誌
logger = setup_logging()

# 注意：請確保 Ollama 伺服器已啟動
# 在終端機運行：ollama serve
logger.info("文件摘要與比較系統啟動")

# 初始化模型
logger.info("初始化 Ollama 模型: deepseek-r1:8b")
llm = OllamaLLM(model="deepseek-r1:8b", temperature=0)

# 定義兩個文件內容 (示例)
doc1 = """產品X是一款手機，具有5吋螢幕、64GB儲存空間和1200萬畫素相機。電池容量3000mAh，支援快充。"""
doc2 = """產品Y是一款手機，配備6吋螢幕、128GB儲存以及1600萬畫素相機。電池3500mAh，並具備快充與無線充電功能。"""
logger.info("已載入要分析的文件內容")
logger.info(f"文件1: {doc1}")
logger.info(f"文件2: {doc2}")

try:
    # 要求模型分別總結兩份文件的重點
    logger.info("開始處理文件摘要")
    print("正在連接 Ollama 伺服器...")
    summary_template = "請幫我摘要以下文件的主要內容：\n{}"
    
    logger.info("處理文件1摘要")
    summary1 = llm.invoke(summary_template.format(doc1))
    logger.info(f"文件1摘要結果: {summary1}")
    
    logger.info("處理文件2摘要")
    summary2 = llm.invoke(summary_template.format(doc2))
    logger.info(f"文件2摘要結果: {summary2}")

    print("文件1摘要:", summary1)
    print("文件2摘要:", summary2)

    # 讓模型比較兩份摘要/文件的異同
    logger.info("開始比較兩份文件的異同")
    compare_prompt = f"文件1的主要內容: {summary1}\n文件2的主要內容: {summary2}\n請比較上述兩者的差異與共同點。"
    comparison = llm.invoke(compare_prompt)
    logger.info(f"比較結果: {comparison}")

    print("比較結果:\n", comparison)
    logger.info("文件分析完成")
    
except Exception as e:
    error_msg = f"連接錯誤: {e}"
    logger.error(error_msg)
    print(error_msg)
    print("\n如果無法連接 Ollama 伺服器，您可能需要：")
    print("1. 安裝並啟動 Ollama: https://ollama.com/download")
    print("2. 運行 'ollama pull deepseek-r1:8b' 下載模型")
    print("3. 如果在 NTU 環境無法使用，考慮使用 OpenAI 模型替代")
    logger.info("程式因錯誤終止")
finally:
    logger.info("文件摘要與比較系統結束")
