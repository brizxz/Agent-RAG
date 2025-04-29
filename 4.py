import logging
import os
import datetime
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM

# 配置日誌
def setup_logging(log_dir="logs"):
    """設置日誌系統"""
    # 確保日誌目錄存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日誌文件名（使用當前日期和時間）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"search_agent_{timestamp}.log")
    
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
logger.info("搜尋代理系統啟動")

# 初始化LLM
logger.info("初始化 Ollama 模型: deepseek-r1:8b")
llm = OllamaLLM(model="deepseek-r1:8b", temperature=0)

# 載入搜尋和網頁請求工具（需要提供 SerpAPI 金鑰）
try:
    # 只使用 serpapi，移除危險的 requests_all 工具
    logger.info("載入 SerpAPI 搜尋工具")
    tools = load_tools(["serpapi"], serpapi_api_key="YOUR_SERPAPI_API_KEY")
    logger.info("搜尋工具載入成功")
    
    # 以下是如何安全使用 requests_all 的方式（如果真的需要的話）
    # tools = load_tools(["serpapi", "requests_all"], 
    #                    serpapi_api_key="YOUR_SERPAPI_API_KEY",
    #                    allow_dangerous_tools=True)

    # 建立Agent
    logger.info("初始化搜尋代理")
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    logger.info("搜尋代理初始化成功")

    # 詢問一個需要深度研究的問題
    query = "量子纏結的基本原理是什麼？請提供相關論文或權威資料的出處。"
    logger.info(f"設定搜尋查詢: {query}")

    print("正在連接 Ollama 伺服器...")
    logger.info("開始執行查詢")
    result = agent.invoke(query)  # 使用 invoke 替代 run
    logger.info("查詢完成")
    logger.info(f"查詢結果: {result}")
    print(result)
    
except Exception as e:
    error_msg = f"錯誤: {e}"
    logger.error(error_msg)
    print(error_msg)
    print("\n如果無法連接 Ollama 伺服器，您可能需要：")
    print("1. 安裝並啟動 Ollama: https://ollama.com/download")
    print("2. 運行 'ollama pull deepseek-r1:8b' 下載模型")
    print("3. 如果在 NTU 環境無法使用，考慮使用 OpenAI 模型替代")
    logger.info("程式因錯誤終止")
finally:
    logger.info("搜尋代理系統結束")
