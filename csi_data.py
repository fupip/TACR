import pandas as pd
import baostock as bs
import akshare as ak

stock_list = ['000338.SZ', '000538.SZ', '600004.SH', '600015.SH', '600016.SH', '600030.SH',
    '600061.SH', '600176.SH', '600177.SH', '600183.SH', '600276.SH', '600340.SH',
    '600346.SH', '600390.SH', '600406.SH', '600487.SH', '600516.SH', '600547.SH',
    '600570.SH', '600588.SH', '600606.SH', '600637.SH', '600760.SH', '600809.SH',
    '600848.SH', '600867.SH', '601009.SH', '601111.SH', '601169.SH', '601919.SH',]


def convert_baostock_data(stock_list):
    bao_stock_dict = {}
    for stock_code in stock_list:
        fs = stock_code.split('.')
        bao_stock_dict[stock_code] = fs[1].lower() + '.' + fs[0]
    return bao_stock_dict

bao_stock_dict = convert_baostock_data(stock_list)

def get_stock_data(stock_code):
    file_path = f"datasets/{stock_code}_day.csv"
    df = pd.read_csv(file_path)
    df["tic"] = stock_code.replace("SH","SS")
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.dayofweek
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    # df = df[df["date"].isin(pd.date_range(start_date,end_date))]
    df = df.reset_index(drop=True)
    return df

def save_stock_data(stock_code,start_date,end_date):
    global bao_stock_dict
    bao_stock_code = bao_stock_dict[stock_code]
    # bs.login()
    rs = bs.query_history_k_data_plus(
            code=bao_stock_code,
            fields="date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",
        )
    print("rs code",rs.error_code)
    
    # while rs.error_code == '0' and rs.next():
    csv_file_path = f"datasets/{stock_code}_day.csv"
    if rs.error_code == '0':
        df = rs.get_data()
        df.to_csv(csv_file_path,index=False)
        print(f"save {stock_code} data to {csv_file_path} Done")
        
        
    # bs.logout()
def get_stock_data_akshare(stock_code, start_date, end_date):
    """
    使用akshare获取A股股票数据
    
    Args:
        stock_code: 股票代码，如 '000338.SZ'
        start_date: 开始日期，如 '20090101'
        end_date: 结束日期，如 '20220101'
    
    Returns:
        DataFrame: 包含股票数据的DataFrame
    """
    
    
    # 转换股票代码格式
    code = stock_code.split('.')[0]
    market = stock_code.split('.')[1].lower()
    
    try:
        # 获取股票数据
        df = ak.stock_zh_a_hist(symbol=code, period="daily", 
                               start_date=start_date, end_date=end_date,
                               adjust="qfq")
        
        # 重命名列
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high', 
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        })
        
        # 添加股票代码列
        df['tic'] = stock_code.replace("SH","SS")
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])
        df['day'] = df['date'].dt.dayofweek
        df['date'] = df['date'].dt.strftime("%Y-%m-%d")
        
        # 保存数据
        file_path = f"datasets/{stock_code}_day.csv"
        df.to_csv(file_path, index=False)
        print(f"save {stock_code} data to {file_path} Done")
        
        return df
        
    except Exception as e:
        print(f"get {stock_code} data failed: {str(e)}")
        return None



def get_csi_stock_data():
    df_list = []
    for stock_code in stock_list:
        df = get_stock_data(stock_code)
        df_list.append(df)
    data_df = pd.concat(df_list)
    data_df = data_df.reset_index(drop=True)
    data_df = data_df.sort_values(['date','tic'],ignore_index=True)
    data_df.to_csv("datasets/csi_day.csv",index=False)
    return data_df

#


if __name__ == "__main__":
    # df = get_stock_data('000338.SZ',start_date="2009-01-01",end_date="2022-01-01")
    
    print(bao_stock_dict)
    
    
    # bs.login()
    # for stock_code in stock_list:
    #     save_stock_data(stock_code,start_date="2009-01-01",end_date="2025-05-20")
        
    # bs.logout()
    
    data_df = get_csi_stock_data()
    print(len(data_df))
    
    
    
