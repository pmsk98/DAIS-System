import os
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler


from quantylab.rltrader import settings


COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']



COLUMNS_TRAINING_DATA_V1 = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V1_1 = COLUMNS_TRAINING_DATA_V1 + [
    'inst_lastinst_ratio', 'frgn_lastfrgn_ratio',
    'inst_ma5_ratio', 'frgn_ma5_ratio',
    'inst_ma10_ratio', 'frgn_ma10_ratio',
    'inst_ma20_ratio', 'frgn_ma20_ratio',
    'inst_ma60_ratio', 'frgn_ma60_ratio',
    'inst_ma120_ratio', 'frgn_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V2 = ['per', 'pbr', 'roe'] + COLUMNS_TRAINING_DATA_V1 + [
    'market_kospi_ma5_ratio', 'market_kospi_ma20_ratio', 
    'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio', 
    'bond_k3y_ma5_ratio', 'bond_k3y_ma20_ratio', 
    'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V3 = COLUMNS_TRAINING_DATA_V2 + [
    'ind', 'ind_diff', 'ind_ma5', 'ind_ma10', 'ind_ma20', 'ind_ma60', 'ind_ma120',
    'inst', 'inst_diff', 'inst_ma5', 'inst_ma10', 'inst_ma20', 'inst_ma60', 'inst_ma120',
    'foreign', 'foreign_diff', 'foreign_ma5', 'foreign_ma10', 'foreign_ma20', 
    'foreign_ma60', 'foreign_ma120',
]
COLUMNS_TRAINING_DATA_V3 = list(map(
    lambda x: x if x != 'close_lastclose_ratio' else 'diffratio', COLUMNS_TRAINING_DATA_V3))

COLUMNS_TRAINING_DATA_V4 = [
    'diffratio', 'high_close_ratio', 'low_close_ratio', 'open_lastclose_ratio', 
    'volume_lastvolume_ratio', 'trans_price_exp', 'trans_price_exp_ma5', 
    'close_ma5_ratio', 'close_ma10_ratio', 'close_ma20_ratio', 'close_ma60_ratio', 'close_ma120_ratio',
    'volume_ma5_ratio', 'volume_ma10_ratio', 'volume_ma20_ratio', 'volume_ma60_ratio', 'volume_ma120_ratio',
    'close_ubb_ratio', 'close_lbb_ratio', 'macd_signal_ratio', 'rsi', 
    'buy_strength_ma5_ratio', 'sell_strength_ma5_ratio', 'prevvalid_cnt',
    'eps_krx', 'bps_krx', 'per_krx', 'pbr_krx', 'roe_krx', 'dps_krx', 'dyr_krx', 'marketcap',
    'ind', 'ind_diff', 'ind_ma5', 'ind_ma10', 'ind_ma20', 'ind_ma60', 'ind_ma120',
    'inst', 'inst_diff', 'inst_ma5', 'inst_ma10', 'inst_ma20', 'inst_ma60', 'inst_ma120',
    'foreign', 'foreign_diff', 'foreign_ma5', 'foreign_ma10', 'foreign_ma20', 'foreign_ma60', 'foreign_ma120',
    'bal_rto', 'bal_rto_diff', 'bal_rto_ma5', 'bal_rto_ma10', 'bal_rto_ma20', 'bal_rto_ma60', 'bal_rto_ma120',
    'short_ratio', 'short_ratio_diff', 'short_ratio_ma5', 'short_ratio_ma10', 'short_ratio_ma20', 'short_ratio_ma60', 'short_ratio_ma120',
    'market_kospi_diffratio', 'market_kospi_ma5_ratio', 'market_kospi_ma20_ratio', 'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio',
    'market_kosdaq_diffratio', 'market_kosdaq_ma5_ratio', 'market_kosdaq_ma20_ratio', 'market_kosdaq_ma60_ratio', 'market_kosdaq_ma120_ratio',
    'market_kospi_volume_diffratio', 'market_kospi_volume_ma5_ratio', 'market_kospi_volume_ma20_ratio', 'market_kospi_volume_ma60_ratio', 'market_kospi_volume_ma120_ratio',
    'market_kosdaq_volume_diffratio', 'market_kosdaq_volume_ma5_ratio', 'market_kosdaq_volume_ma20_ratio', 'market_kosdaq_volume_ma60_ratio', 'market_kosdaq_volume_ma120_ratio',
    'fmarket_dji_diffratio', 'fmarket_dji_ma5_ratio', 'fmarket_dji_ma20_ratio', 'fmarket_dji_ma60_ratio', 'fmarket_dji_ma120_ratio',
    'fmarket_ni225_diffratio', 'fmarket_ni225_ma5_ratio', 'fmarket_ni225_ma20_ratio', 'fmarket_ni225_ma60_ratio', 'fmarket_ni225_ma120_ratio',
    'fmarket_hsi_diffratio', 'fmarket_hsi_ma5_ratio', 'fmarket_hsi_ma20_ratio', 'fmarket_hsi_ma60_ratio', 'fmarket_hsi_ma120_ratio',
    'fmarket_dji_volume_diffratio', 'fmarket_dji_volume_ma5_ratio', 'fmarket_dji_volume_ma20_ratio', 'fmarket_dji_volume_ma60_ratio', 'fmarket_dji_volume_ma120_ratio',
    'fmarket_ni225_volume_diffratio', 'fmarket_ni225_volume_ma5_ratio', 'fmarket_ni225_volume_ma20_ratio', 'fmarket_ni225_volume_ma60_ratio', 'fmarket_ni225_volume_ma120_ratio',
    'fmarket_hsi_volume_diffratio', 'fmarket_hsi_volume_ma5_ratio', 'fmarket_hsi_volume_ma20_ratio', 'fmarket_hsi_volume_ma60_ratio', 'fmarket_hsi_volume_ma120_ratio',
    'bond_k3y_diffratio', 'bond_k3y_ma5_ratio', 'bond_k3y_ma20_ratio', 'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio',
    'bond_k3y_volume_diffratio', 'bond_k3y_volume_ma5_ratio', 'bond_k3y_volume_ma20_ratio', 'bond_k3y_volume_ma60_ratio', 'bond_k3y_volume_ma120_ratio',
    'interestrates_base_diffratio', 'interestrates_base_ma5_ratio', 'interestrates_base_ma20_ratio', 'interestrates_base_ma60_ratio', 'interestrates_base_ma120_ratio', 'interestrates_base',
    'interestrates_us_target_diffratio', 'interestrates_us_target_ma5_ratio', 'interestrates_us_target_ma20_ratio', 'interestrates_us_target_ma60_ratio', 'interestrates_us_target_ma120_ratio', 'interestrates_us_target',
    'interestrates_us_10y_diffratio', 'interestrates_us_10y_ma5_ratio', 'interestrates_us_10y_ma20_ratio', 'interestrates_us_10y_ma60_ratio', 'interestrates_us_10y_ma120_ratio', 'interestrates_us_10y',
    'interestrates_us_30y_diffratio', 'interestrates_us_30y_ma5_ratio', 'interestrates_us_30y_ma20_ratio', 'interestrates_us_30y_ma60_ratio', 'interestrates_us_30y_ma120_ratio', 'interestrates_us_30y',
    'commodity_aluminum_diffratio', 'commodity_aluminum_ma5_ratio', 'commodity_aluminum_ma20_ratio', 'commodity_aluminum_ma60_ratio', 'commodity_aluminum_ma120_ratio',
    'commodity_cocoa_diffratio', 'commodity_cocoa_ma5_ratio', 'commodity_cocoa_ma20_ratio', 'commodity_cocoa_ma60_ratio', 'commodity_cocoa_ma120_ratio',
    'commodity_coffee_diffratio', 'commodity_coffee_ma5_ratio', 'commodity_coffee_ma20_ratio', 'commodity_coffee_ma60_ratio', 'commodity_coffee_ma120_ratio',
    'commodity_copper_diffratio', 'commodity_copper_ma5_ratio', 'commodity_copper_ma20_ratio', 'commodity_copper_ma60_ratio', 'commodity_copper_ma120_ratio',
    'commodity_corn_diffratio', 'commodity_corn_ma5_ratio', 'commodity_corn_ma20_ratio', 'commodity_corn_ma60_ratio', 'commodity_corn_ma120_ratio',
    'commodity_cotton_diffratio', 'commodity_cotton_ma5_ratio', 'commodity_cotton_ma20_ratio', 'commodity_cotton_ma60_ratio', 'commodity_cotton_ma120_ratio',
    'commodity_gold_domestic_diffratio', 'commodity_gold_domestic_ma5_ratio', 'commodity_gold_domestic_ma20_ratio', 'commodity_gold_domestic_ma60_ratio', 'commodity_gold_domestic_ma120_ratio',
    'commodity_gold_world_diffratio', 'commodity_gold_world_ma5_ratio', 'commodity_gold_world_ma20_ratio', 'commodity_gold_world_ma60_ratio', 'commodity_gold_world_ma120_ratio',
    'commodity_heating_oil_diffratio', 'commodity_heating_oil_ma5_ratio', 'commodity_heating_oil_ma20_ratio', 'commodity_heating_oil_ma60_ratio', 'commodity_heating_oil_ma120_ratio',
    'commodity_iron_diffratio', 'commodity_iron_ma5_ratio', 'commodity_iron_ma20_ratio', 'commodity_iron_ma60_ratio', 'commodity_iron_ma120_ratio',
    'commodity_lead_diffratio', 'commodity_lead_ma5_ratio', 'commodity_lead_ma20_ratio', 'commodity_lead_ma60_ratio', 'commodity_lead_ma120_ratio',
    'commodity_lumber_diffratio', 'commodity_lumber_ma5_ratio', 'commodity_lumber_ma20_ratio', 'commodity_lumber_ma60_ratio', 'commodity_lumber_ma120_ratio',
    'commodity_natural_gas_diffratio', 'commodity_natural_gas_ma5_ratio', 'commodity_natural_gas_ma20_ratio', 'commodity_natural_gas_ma60_ratio', 'commodity_natural_gas_ma120_ratio',
    'commodity_nickel_diffratio', 'commodity_nickel_ma5_ratio', 'commodity_nickel_ma20_ratio', 'commodity_nickel_ma60_ratio', 'commodity_nickel_ma120_ratio',
    'commodity_oil_diesel_diffratio', 'commodity_oil_diesel_ma5_ratio', 'commodity_oil_diesel_ma20_ratio', 'commodity_oil_diesel_ma60_ratio', 'commodity_oil_diesel_ma120_ratio',
    'commodity_oil_gasoline_diffratio', 'commodity_oil_gasoline_ma5_ratio', 'commodity_oil_gasoline_ma20_ratio', 'commodity_oil_gasoline_ma60_ratio', 'commodity_oil_gasoline_ma120_ratio',
    'commodity_oil_wti_diffratio', 'commodity_oil_wti_ma5_ratio', 'commodity_oil_wti_ma20_ratio', 'commodity_oil_wti_ma60_ratio', 'commodity_oil_wti_ma120_ratio',
    'commodity_orange_juice_diffratio', 'commodity_orange_juice_ma5_ratio', 'commodity_orange_juice_ma20_ratio', 'commodity_orange_juice_ma60_ratio', 'commodity_orange_juice_ma120_ratio',
    'commodity_palladium_diffratio', 'commodity_palladium_ma5_ratio', 'commodity_palladium_ma20_ratio', 'commodity_palladium_ma60_ratio', 'commodity_palladium_ma120_ratio',
    'commodity_platinum_diffratio', 'commodity_platinum_ma5_ratio', 'commodity_platinum_ma20_ratio', 'commodity_platinum_ma60_ratio', 'commodity_platinum_ma120_ratio',
    'commodity_rice_diffratio', 'commodity_rice_ma5_ratio', 'commodity_rice_ma20_ratio', 'commodity_rice_ma60_ratio', 'commodity_rice_ma120_ratio',
    'commodity_silver_diffratio', 'commodity_silver_ma5_ratio', 'commodity_silver_ma20_ratio', 'commodity_silver_ma60_ratio', 'commodity_silver_ma120_ratio',
    'commodity_soybean_diffratio', 'commodity_soybean_ma5_ratio', 'commodity_soybean_ma20_ratio', 'commodity_soybean_ma60_ratio', 'commodity_soybean_ma120_ratio',
    'commodity_soybean_gourd_diffratio', 'commodity_soybean_gourd_ma5_ratio', 'commodity_soybean_gourd_ma20_ratio', 'commodity_soybean_gourd_ma60_ratio', 'commodity_soybean_gourd_ma120_ratio',
    'commodity_soybean_milk_diffratio', 'commodity_soybean_milk_ma5_ratio', 'commodity_soybean_milk_ma20_ratio', 'commodity_soybean_milk_ma60_ratio', 'commodity_soybean_milk_ma120_ratio',
    'commodity_sugar11_diffratio', 'commodity_sugar11_ma5_ratio', 'commodity_sugar11_ma20_ratio', 'commodity_sugar11_ma60_ratio', 'commodity_sugar11_ma120_ratio',
    'commodity_tin_diffratio', 'commodity_tin_ma5_ratio', 'commodity_tin_ma20_ratio', 'commodity_tin_ma60_ratio', 'commodity_tin_ma120_ratio',
    'commodity_wheat_diffratio', 'commodity_wheat_ma5_ratio', 'commodity_wheat_ma20_ratio', 'commodity_wheat_ma60_ratio', 'commodity_wheat_ma120_ratio',
    'commodity_zinc_diffratio', 'commodity_zinc_ma5_ratio', 'commodity_zinc_ma20_ratio', 'commodity_zinc_ma60_ratio', 'commodity_zinc_ma120_ratio',
    'gsci_diffratio', 'gsci_ma5_ratio', 'gsci_ma20_ratio', 'gsci_ma60_ratio', 'gsci_ma120_ratio',
    'fx_usdkrw_diffratio', 'fx_usdkrw_ma5_ratio', 'fx_usdkrw_ma20_ratio', 'fx_usdkrw_ma60_ratio', 'fx_usdkrw_ma120_ratio',
    'fx_eurkrw_diffratio', 'fx_eurkrw_ma5_ratio', 'fx_eurkrw_ma20_ratio', 'fx_eurkrw_ma60_ratio', 'fx_eurkrw_ma120_ratio',
    'fx_jpykrw_diffratio', 'fx_jpykrw_ma5_ratio', 'fx_jpykrw_ma20_ratio', 'fx_jpykrw_ma60_ratio', 'fx_jpykrw_ma120_ratio',
    'fx_cnykrw_diffratio', 'fx_cnykrw_ma5_ratio', 'fx_cnykrw_ma20_ratio', 'fx_cnykrw_ma60_ratio', 'fx_cnykrw_ma120_ratio',
    'fx_hkdkrw_diffratio', 'fx_hkdkrw_ma5_ratio', 'fx_hkdkrw_ma20_ratio', 'fx_hkdkrw_ma60_ratio', 'fx_hkdkrw_ma120_ratio',
    'dx_diffratio', 'dx_ma5_ratio', 'dx_ma20_ratio', 'dx_ma60_ratio', 'dx_ma120_ratio',
    'dx_volume_diffratio', 'dx_volume_ma5_ratio', 'dx_volume_ma20_ratio', 'dx_volume_ma60_ratio', 'dx_volume_ma120_ratio',
    'bdi_diffratio', 'bdi_ma5_ratio', 'bdi_ma20_ratio', 'bdi_ma60_ratio', 'bdi_ma120_ratio',
    'sox_diffratio', 'sox_ma5_ratio', 'sox_ma20_ratio', 'sox_ma60_ratio', 'sox_ma120_ratio',
    'vix_diffratio', 'vix_ma5_ratio', 'vix_ma20_ratio', 'vix_ma60_ratio', 'vix_ma120_ratio',
    'msci_world_diffratio', 'msci_world_ma5_ratio', 'msci_world_ma20_ratio', 'msci_world_ma60_ratio', 'msci_world_ma120_ratio',
    'msci_acwi_diffratio', 'msci_acwi_ma5_ratio', 'msci_acwi_ma20_ratio', 'msci_acwi_ma60_ratio', 'msci_acwi_ma120_ratio',
    'msci_em_diffratio', 'msci_em_ma5_ratio', 'msci_em_ma20_ratio', 'msci_em_ma60_ratio', 'msci_em_ma120_ratio',
    'msci_korea_diffratio', 'msci_korea_ma5_ratio', 'msci_korea_ma20_ratio', 'msci_korea_ma60_ratio', 'msci_korea_ma120_ratio',
    'msci_usa_diffratio', 'msci_usa_ma5_ratio', 'msci_usa_ma20_ratio', 'msci_usa_ma60_ratio', 'msci_usa_ma120_ratio',
    'msci_china_diffratio', 'msci_china_ma5_ratio', 'msci_china_ma20_ratio', 'msci_china_ma60_ratio', 'msci_china_ma120_ratio',
    'msci_japan_diffratio', 'msci_japan_ma5_ratio', 'msci_japan_ma20_ratio', 'msci_japan_ma60_ratio', 'msci_japan_ma120_ratio',
    'msci_hongkong_diffratio', 'msci_hongkong_ma5_ratio', 'msci_hongkong_ma20_ratio', 'msci_hongkong_ma60_ratio', 'msci_hongkong_ma120_ratio',
    'msci_uk_diffratio', 'msci_uk_ma5_ratio', 'msci_uk_ma20_ratio', 'msci_uk_ma60_ratio', 'msci_uk_ma120_ratio',
    'msci_france_diffratio', 'msci_france_ma5_ratio', 'msci_france_ma20_ratio', 'msci_france_ma60_ratio', 'msci_france_ma120_ratio',
    'msci_germany_diffratio', 'msci_germany_ma5_ratio', 'msci_germany_ma20_ratio', 'msci_germany_ma60_ratio', 'msci_germany_ma120_ratio',
]

def preprocess(data, ver='v1'):
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        data[f'close_ma{window}'] = data['close'].rolling(window).mean()
        data[f'volume_ma{window}'] = data['volume'].rolling(window).mean()
        data[f'close_ma{window}_ratio'] = \
            (data['close'] - data[f'close_ma{window}']) / data[f'close_ma{window}']
        data[f'volume_ma{window}_ratio'] = \
            (data['volume'] - data[f'volume_ma{window}']) / data[f'volume_ma{window}']
        
    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = \
        (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = \
        (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (
        (data['volume'][1:].values - data['volume'][:-1].values) 
        / data['volume'][:-1].replace(to_replace=0, method='ffill')\
            .replace(to_replace=0, method='bfill').values
    )

    if ver == 'v1.1':
        for window in windows:
            data[f'inst_ma{window}'] = data['close'].rolling(window).mean()
            data[f'frgn_ma{window}'] = data['volume'].rolling(window).mean()
            data[f'inst_ma{window}_ratio'] = \
                (data['close'] - data[f'inst_ma{window}']) / data[f'inst_ma{window}']
            data[f'frgn_ma{window}_ratio'] = \
                (data['volume'] - data[f'frgn_ma{window}']) / data[f'frgn_ma{window}']
        data['inst_lastinst_ratio'] = np.zeros(len(data))
        data.loc[1:, 'inst_lastinst_ratio'] = (
            (data['inst'][1:].values - data['inst'][:-1].values)
            / data['inst'][:-1].replace(to_replace=0, method='ffill')\
                .replace(to_replace=0, method='bfill').values
        )
        data['frgn_lastfrgn_ratio'] = np.zeros(len(data))
        data.loc[1:, 'frgn_lastfrgn_ratio'] = (
            (data['frgn'][1:].values - data['frgn'][:-1].values)
            / data['frgn'][:-1].replace(to_replace=0, method='ffill')\
                .replace(to_replace=0, method='bfill').values
        )

    return data


def load_data(code, date_from, date_to, ver='v2'):
    if ver in ['v3', 'v4']:
        return load_data_v3_v4(code, date_from, date_to, ver)
    elif ver in ['v5'] :
        return load_data_sample(code,date_from,date_to,ver)
    else :
        pass

    header = None if ver == 'v1' else 0
    df = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, f'{code}.csv'),
        thousands=',', header=header, converters={'date': lambda x: str(x)})

    if ver == 'v1':
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # 데이터 전처리
    df = preprocess(df)
    
    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.fillna(method='ffill').reset_index(drop=True)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = None
    if ver == 'v1':
        training_data = df[COLUMNS_TRAINING_DATA_V1]
    elif ver == 'v1.1':
        training_data = df[COLUMNS_TRAINING_DATA_V1_1]
    elif ver == 'v2':
        df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = df[COLUMNS_TRAINING_DATA_V2]
        training_data = training_data.apply(np.tanh)
    else:
        raise Exception('Invalid version.')
    
    return chart_data, training_data


def load_data_v3_v4(code, date_from, date_to, ver):
    columns = None
    if ver == 'v3':
        columns = COLUMNS_TRAINING_DATA_V3
    elif ver == 'v4':
        columns = COLUMNS_TRAINING_DATA_V4

    # 시장 데이터
    df_marketfeatures = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, 'marketfeatures.csv'), 
        thousands=',', header=0, converters={'date': lambda x: str(x)})
    
    # 종목 데이터
    df_stockfeatures = None
    for filename in os.listdir(os.path.join(settings.BASE_DIR, 'data', ver)):
        if filename.startswith(code):
            df_stockfeatures = pd.read_csv(
                os.path.join(settings.BASE_DIR, 'data', ver, filename), 
                thousands=',', header=0, converters={'date': lambda x: str(x)})
            break

    # 시장 데이터와 종목 데이터 합치기
    df = pd.merge(df_stockfeatures, df_marketfeatures, on='date', how='left', suffixes=('', '_dup'))
    df = df.drop(df.filter(regex='_dup$').columns.tolist(), axis=1)

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.fillna(method='ffill').reset_index(drop=True)

    # 데이터 조정
    df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = df[columns]

    return chart_data, training_data


def load_data_sample(code,date_from,date_to,ver):
    for filename in os.listdir(os.path.join(settings.BASE_DIR,'data',ver)):
        if filename.startswith(code):
            df_stockfeatures = None
            df_stockfeatures = pd.read_csv(os.path.join(settings.BASE_DIR,'data',ver,filename),
                                            thousands=',',header=0,converters={'date': lambda x: str(x)})
            break
    df = df_stockfeatures
    
    #momentum indicators
    ADX=talib.ADX(df.high,df.low,df.close,timeperiod=14)

    ADXR=talib.ADXR(df.high,df.low,df.close,timeperiod=14)
    
    APO=talib.APO(df.close,fastperiod=12,slowperiod=26,matype=0)
    
    aroondown,aroonup =talib.AROON(df.high, df.low, timeperiod=14)
    
    AROONOSC=talib.AROONOSC(df.high,df.low,timeperiod=14)
    
    BOP=talib.BOP(df.open,df.high,df.low,df.close)
    
    CCI=talib.CCI(df.high,df.low,df.close,timeperiod=14)
    
    CMO=talib.CMO(df.close,timeperiod=14)
    
    DX=talib.DX(df.high,df.low,df.close,timeperiod=14)
    
    macd, macdsignal, macdhist = talib.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)
    
    ma_macd, ma_macdsignal, ma_macdhist = talib.MACDEXT(df.close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    
    fix_macd,fix_macdsignal, fix_macdhist = talib.MACDFIX(df.close, signalperiod=9)
    
    MFI=talib.MFI(df.high, df.low,df.close, df.volume, timeperiod=14)
    
    MINUS_DI=talib.MINUS_DI(df.high, df.low, df.close, timeperiod=14)
    
    MINUS_DM=talib. MINUS_DM(df.high, df.low, timeperiod=14)
    
    MOM=talib.MOM(df.close,timeperiod=10)
    
    PLUS_DM=talib.PLUS_DM(df.high,df.low,timeperiod=14)
    
    PPO=talib.PPO(df.close, fastperiod=12, slowperiod=26, matype=0)
    
    ROC=talib.ROC(df.close,timeperiod=10)
    
    ROCP=talib.ROCP(df.close,timeperiod=10)
    
    ROCR=talib.ROCR(df.close,timeperiod=10)
    
    ROCR100=talib.ROCR100(df.close,timeperiod=10)
    
    RSI=talib.RSI(df.close,timeperiod=14)
    
    slowk, slowd = talib.STOCH(df.high, df.low, df.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    fastk, fastd = talib.STOCHF(df.high, df.low, df.close, fastk_period=5, fastd_period=3, fastd_matype=0)
    
    TRIX=talib.TRIX(df.close,timeperiod=30)
    
    ULTOSC=talib.ULTOSC(df.high,df.low,df.close,timeperiod1=7,timeperiod2=14,timeperiod3=28)
    
    WILLR=talib.WILLR(df.high,df.low,df.close,timeperiod=14)
    
    #pattern recognition
    two_crows = talib.CDL2CROWS(df.open,df.high,df.low,df.close)
    
    three_black_crows = talib.CDL3BLACKCROWS(df.open,df.high,df.low,df.close)
    
    three_inside = talib.CDL3INSIDE(df.open,df.high,df.low,df.close)
    
    three_line_strike = talib.CDL3LINESTRIKE(df.open,df.high,df.low,df.close)
    
    three_outside = talib.CDL3OUTSIDE(df.open,df.high,df.low,df.close)
    
    three_star = talib.CDL3STARSINSOUTH(df.open,df.high,df.low,df.close)
    
    three_advance_white = talib.CDL3WHITESOLDIERS(df.open,df.high,df.low,df.close)
    
    abandoned_baby = talib.CDLABANDONEDBABY(df.open,df.high,df.low,df.close,penetration=0)

    
    df['ADX']=ADX
    df['ADXR']=ADXR
    df['APO']=APO
    df['aroondown']=aroondown
    df['aroonup']=aroonup
    df['AROONOSC']=AROONOSC
    df['BOP']=BOP
    df['CCI']=CCI
    df['CMO']=CMO
    df['DX']=DX
    df['MACD']=macd
    df['macdsignal']=macdsignal
    df['macdhist']=macdhist
    df['ma_macd']=ma_macd
    df['ma_macdsignal']=ma_macdsignal
    df['ma_macdhist']=ma_macdhist
    df['fix_macd']=fix_macd
    df['fix_macdsignal']=fix_macdsignal
    df['fix_macdhist']=fix_macdhist
    df['MFI']=MFI
    df['MINUS_DI']=MINUS_DI
    df['MINUS_DM']=MINUS_DM
    df['MOM']=MOM
    df['PLUS_DM']=PLUS_DM
    df['PPO']=PPO
    df['ROC']=ROC
    df['ROCP']=ROCP
    df['ROCR']=ROCR
    df['ROCR100']=ROCR100
    df['RSI']=RSI
    df['slowk']=slowk
    df['slowd']=slowd
    df['fastk']=fastk
    df['fastd']=fastd
    df['TRIX']=TRIX
    df['ULTOSC']=ULTOSC
    df['WILLR']=WILLR
    df['two_crows']= two_crows
    df['three_black_crows'] = three_black_crows
    df['three_inside'] = three_inside
    df['three_line_strike'] = three_line_strike
    df['three_outside'] = three_outside
    df['three_star'] = three_star
    df['three_advance_white'] = three_advance_white
    df['abandoned_baby'] = abandoned_baby
    
   
    
    df = df.sort_values(by='date').reset_index(drop=True)
    
    df  = df.drop(['Unnamed: 0'],axis=1)
    
    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.fillna(method='ffill').reset_index(drop=True)
    df['date'] =df['date'].astype('float')
    
    minmax=MinMaxScaler()   

 
    minmax_feature=df[['ADX','ADXR','APO','aroondown','aroonup','AROONOSC','BOP','CCI','CMO','DX','MACD','macdsignal','ma_macd','ma_macdsignal','ma_macdhist','ma_macdhist','fix_macd','fix_macdsignal','fix_macdhist','MFI','MINUS_DI','MINUS_DM','MOM','PLUS_DM','PPO','ROC','ROCP','ROCR','ROCR100','RSI','slowk','slowd','fastk','fastd','TRIX','ULTOSC','WILLR','two_crows','three_black_crows','three_inside','three_line_strike','three_outside','three_star','three_advance_white','abandoned_baby']]
    minmax_feature=minmax.fit_transform(minmax_feature)
    df[['ADX','ADXR','APO','aroondown','aroonup','AROONOSC','BOP','CCI','CMO','DX','MACD','macdsignal','ma_macd','ma_macdsignal','ma_macdhist','ma_macdhist','fix_macd','fix_macdsignal','fix_macdhist','MFI','MINUS_DI','MINUS_DM','MOM','PLUS_DM','PPO','ROC','ROCP','ROCR','ROCR100','RSI','slowk','slowd','fastk','fastd','TRIX','ULTOSC','WILLR','two_crows','three_black_crows','three_inside','three_line_strike','three_outside','three_star','three_advance_white','abandoned_baby']]=minmax_feature
        
    #차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]
    
    training_data = df
    
    return chart_data,training_data



