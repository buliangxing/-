import akshare as ak
import pandas as pd

# 粘贴上面修改后的get_stock_basic_info函数
def get_stock_basic_info() -> pd.DataFrame:
    stock_info = ak.stock_info_a_code_name()
    stock_info.rename(columns={'code': '代码', 'name': '名称'}, inplace=True)
    print(f"✅ 列名映射成功！当前列名：{stock_info.columns.tolist()}")
    return stock_info

# 测试遍历“代码”列
if __name__ == "__main__":
    stock_info = get_stock_basic_info()
    # 遍历前5只股票的代码，验证无KeyError
    for i, code in enumerate(stock_info['代码'].head(5)):
        print(f"第{i+1}只股票代码：{code}，名称：{stock_info['名称'].iloc[i]}")