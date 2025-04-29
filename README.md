# LangChain AI 應用示例

此repository包含四個使用LangChain和Ollama的AI應用示例，展示了不同類型的應用場景。所有程式均使用繁體中文，並基於`deepseek-r1:8b`模型。

## 安裝需求

在運行任何示例前，請確保已安裝以下軟體和套件：

1. Python 3.8+
2. Ollama (https://ollama.com/download)
3. 必要的Python套件：
```bash
pip install langchain langchain_community langchain_ollama faiss-cpu
```

4. 下載所需模型：
```bash
ollama pull deepseek-r1:8b
```

## 程式列表

### 1. 知識庫問答代理 (1.py)

此程式展示了建立一個能夠回答公司相關問題的AI代理系統。

**功能**：
- 建立並使用向量資料庫存儲知識內容
- 提供基於知識庫的問答能力
- 支援簡單計算功能
- 具備對話記憶，能理解上下文

**使用方法**：
```bash
python 1.py
```

**輸出**：
- 控制台輸出
- 日誌文件：logs/qa_agent_[timestamp].log

### 2. 文件摘要與比較 (2.py)

此程式演示如何使用AI進行文件摘要並比較不同文件的異同。

**功能**：
- 提取兩個產品文件的摘要
- 比較兩份文件的異同點
- 生成結構化的比較報告

**使用方法**：
```bash
python 2.py
```

**輸出**：
- 控制台輸出
- 日誌文件：logs/document_compare_[timestamp].log

### 3. AI生成問卷系統 (3.py)

此程式展示了如何使用AI智能生成問卷，收集回答並保存結果。

**功能**：
- 根據用戶指定主題自動生成問卷問題
- 自動增加格式標示(如評分1-5分)
- 將回答存儲為CSV格式
- 支援自定義問題數量與輸出位置

**使用方法**：
```bash
python 3.py
```

**輸出**：
- 控制台互動介面
- CSV格式的問卷結果（預設為 問卷調查結果.csv）

### 4. 網路搜尋代理 (4.py)

此程式示範如何建立一個能夠搜尋網路資訊的AI代理。

**功能**：
- 使用SerpAPI進行網路搜尋
- 回答需要查詢外部資料的複雜問題
- 能夠引用外部資料來源

**使用方法**：
```bash
python 4.py
```

**注意事項**：
- 需要提供有效的SerpAPI金鑰

**輸出**：
- 控制台輸出
- 日誌文件：logs/search_agent_[timestamp].log

## 常見問題

1. **連接錯誤**
   - 確保Ollama服務已經啟動：`ollama serve`
   - 確認模型已經下載：`ollama pull deepseek-r1:8b`

2. **SerpAPI問題**
   - 需要註冊並獲取API金鑰：https://serpapi.com/
   - 在4.py中替換"YOUR_SERPAPI_API_KEY"為實際的API金鑰

3. **CSV格式問題**
   - 如果使用Microsoft Excel開啟CSV文件，可能需要設置UTF-8編碼

## 日誌系統

所有程式（除了3.py的問卷系統外）都實現了完整的日誌系統：

- 日誌保存在`logs`目錄下
- 每個日誌文件命名格式：`[程式類型]_[時間戳].log`
- 日誌記錄程式執行的每個步驟和結果 