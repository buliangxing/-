import akshare as ak

# 唯一目的：查看akshare返回的原始列名
df = ak.stock_info_a_code_name()
print("=== akshare 返回的原始列名 ===")
print(df.columns.tolist())  # 打印所有列名
print("\n=== 前3行原始数据 ===")
print(df.head(3))  # 打印前3行数据