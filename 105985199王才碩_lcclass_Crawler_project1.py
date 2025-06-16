# !pip install wordcloud
# !pip install Pillow
# !pip install jieba
# !pip install emoji
# !pip install SciencePlots
# !pip install matplotlib-venn
#%% import packages
import emoji
# from emoji import UNICODE_EMOJI #The property UNICODE_EMOJI was removed in version 2.0.0 of emoji module.
from emoji import EMOJI_DATA
import wordcloud
from wordcloud import WordCloud
from PIL import ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from matplotlib_venn import venn2
import matplotlib.colors as mcolors
import jieba
import jieba.analyse
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import datetime
from dateutil.relativedelta import relativedelta
import os
import scienceplots
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import seaborn as sns
from scipy import stats
import statsmodels.api as sm 
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from scipy.stats import t

#%% define functions
def is_emoji(s):
    return s in EMOJI_DATA

def set_workspace(worksp_path=None):
    # 如果沒有提供 worksp_path 且 __file__ 未定義，則使用當前工作目錄
    if worksp_path is None:
        try:
            worksp_path = os.path.dirname(os.path.abspath(__file__)) # 使用py檔路徑作為工作目錄
        except NameError:
            worksp_path = os.getcwd() # 使用當前路徑
            print('未能正確取得__file__檔案路徑，設置當前路徑為工作空間。')
    # os.makedirs(worksp_path, exist_ok=True)  # 若路徑不存在，則創建資料夾
    os.chdir(worksp_path)  # 設定工作空間路徑

def plot_rc():
    SIZE_DEFAULT = 14
    SIZE_LARGE = 16
    plt.rc("font", family="DejaVu Sans")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

def load_data_yt(filepath="./source/youtube/Youtube Trending Video - Taiwan & Hong Kong/trending.csv"):
    df_youtrend = pd.read_csv(filepath)
    # 去除不需要的留下所需資料
    dropcolumns = ["thumbnail_url","thumbnail_width",
                   "thumbnail_height","live_status","local_title",
                   "local_description","dimension","definition",
                   "caption","license_status","allowed_region",
                   "blocked_region","dislike","favorite"]
    # 去除重複資料並回傳重複資料集df_dup_yt
    df_youtrend = df_youtrend.drop(columns = dropcolumns)
    df_dup_yt = df_youtrend[df_youtrend.duplicated()] # 存在51107筆重複資料 (6252筆未重複)
    df_youtrend.drop_duplicates(keep='first', inplace=True) #保留重複資料的第一筆
    # 建立年-月類別變數 (datetime %Y-%m) trending-ym
    df_youtrend['trending_time'] = pd.to_datetime(df_youtrend['trending_time'])
    df_youtrend['trending-ym'] = df_youtrend['trending_time'].dt.strftime('%Y-%m')
    
    return df_youtrend, df_dup_yt
    
def load_data_covid(filepath="./source/covid/2023-cdc-Day_Confirmation_Age_County_Gender_19CoV_v1_全國_全區.json"):
    # DoDiag: Date of Diagnosis 個案研判日
    # NwCase: number of New confirmed Cases 新增確診人數
    # CumCas: Cumulative number of confirmed Cases 累計確診人數
    # 7ma_NwCase: Seven-day moving average number of New confirmed Cases 七日移動平均新增確診人數
    df_covidTW = pd.read_json(filepath)
    df_covidTW = df_covidTW.drop(columns=["a02","a03"])
    df_covidTW.columns = ["ID","DoDiag","NwCase","CumCas","7ma_NwCase"] # columns命名
    df_covidTW = df_covidTW.sort_values(by='DoDiag',ascending=True) # 由上至下由舊至新
    
    # 資料僅在新案例診斷發生時的紀錄，無診斷日就會漏掉。為了計算moving standard deviantion，我還是需要將這些無診斷日的資料補回來 (UID寫0)
    # 先用pd.date_range()建構完整日期，再透過df[].fillna(method='ffill') <-backward或 df[].fillna(method='bfill') <-forward 填補缺失值
    start_date = df_covidTW["DoDiag"].min()
    end_date = df_covidTW["DoDiag"].max()
    date_range = pd.DataFrame({'DoDiag':pd.date_range(start=start_date,end=end_date)})
    df_covidTW['DoDiag'] = pd.to_datetime(df_covidTW['DoDiag'])
    df_covidTW = pd.merge(date_range, df_covidTW, on='DoDiag', how='left')
    
    # df_covidTW['ID'].fillna(0, inplace=True)
    df_covidTW['NwCase'].fillna(0, inplace=True) # 在NwCase以0填補缺失值
    df_covidTW['CumCas'] = df_covidTW['CumCas'].fillna(method='ffill') # 在CumCas用上一筆填補缺失值
    # 重新計算(recal) 七日移動平均並且更新至原表格 (rolling(7)移動向後取7步； sum().div(7) 七項相加除以7； round(2) 四捨五入至小數點第二位)
    recal_7ma_NwCase = pd.DataFrame({'7ma_NwCase':df_covidTW['NwCase'].rolling(7).sum().div(7).round(2)})
    recal_7ma_NwCase['7ma_NwCase'].loc[0:5] = df_covidTW['CumCas'].loc[0:5].div(7).round(2) #寫入前五筆空缺值
    df_covidTW['7ma_NwCase'].update(recal_7ma_NwCase['7ma_NwCase']) #耕心原表格
    # 計算七日移動標準差並加入表格 (前五項計算意外麻煩，嘗試在NwCase建立[0]*6 series遇bug，不然就是要用迴圈手動處理)
    # 但迴圈效能差又占行，前六筆七日移動標準差的分析意義不大，就不繼續浪費時間計算了
    cal_7mstd_NwCase = pd.DataFrame({'7mstd_NwCase':df_covidTW['NwCase'].rolling(7).std().round(2)})
    df_covidTW['7mstd_NwCase'] = cal_7mstd_NwCase['7mstd_NwCase']
    
    # 建立delay time series NwCase (時間關係先只做到第4天)
    df_covidTW['delay1_NwCase'] = df_covidTW['NwCase'].shift(1)
    df_covidTW['delay2_NwCase'] = df_covidTW['NwCase'].shift(2)
    df_covidTW['delay3_NwCase'] = df_covidTW['NwCase'].shift(3)
    df_covidTW['delay4_NwCase'] = df_covidTW['NwCase'].shift(4)
    
    # 建立年-月類別變數 (datetime %Y-%m) DoDiag_ym
    df_covidTW['DoDiag_ym'] = df_covidTW['DoDiag'].dt.strftime('%Y-%m')
    return df_covidTW

def get_min_date_max_date(Series_date):
    min_date = Series_date.min()
    min_date = min_date.to_pydatetime()
    min_date = min_date.replace(tzinfo=None) # 時區資訊類似bug，會使後續between不能比較
    max_date = Series_date.max()
    max_date = max_date.to_pydatetime()
    max_date = max_date.replace(tzinfo=None)
    return min_date, max_date

def covid_between_yt_period (df_covidTW,dat_yt):
    df_covidTW['DoDiag'] = pd.to_datetime(df_covidTW['DoDiag'])
    # dat_yt['trending_time'] = pd.to_datetime(dat_yt['trending_time'])
    
    # 用yt資料時間全距篩選出cov19的時間範圍
    start_date, end_date = get_min_date_max_date(dat_yt['trending_time'])
    
    # 篩選 data 中 datetime 落在範圍內的資料
    filtered_covidTW = df_covidTW[df_covidTW['DoDiag'].between(start_date, end_date)]
    filtered_covidTW = filtered_covidTW.reset_index()
    
    # 顯示結果
    return filtered_covidTW

def words_tokenization (df,str_var):
    # 仰賴jieba套件對繁中為主的文本做分詞
    raw_list = df[str_var].tolist()
    # 刪除文檔中的標點符號
    text = "".join(q for q in raw_list)
    punc_set = ('‍','️','　','​','│','’','￼',' ','〈','〉','⋯ ','）','（','•','丶','』','『','”','“',
                '⁉️','丨',',','｀','▽','´','(',')','#','》','《','|','｜','；','，','。','！',
                '：','「','」','…','、','？','【','】','.',':','?',';','!','~','`','+','-','<',
                '>','/','[',']','{','}',"'",'"')
    clean_text = "".join(c for c in text if c not in punc_set)
    # 進行斷詞 (Tokenization)
    tokens = jieba.lcut(clean_text, cut_all=False)
    return tokens

def get_stopwords (stop_dir='./source/stopwords'):
    # 請將所有要使用的stopwords檔案儲存在同一個資料夾stop_dir
    # 目前的context用不太到import nltk 與 from nltk.corpus import stopwords (組合兩邊資料有點小麻煩)
    stop_dir = stop_dir
    txt_files_name = [f for f in os.listdir(stop_dir) if f.endswith('.txt')]
    stopwords = {}    
    for txt_file in txt_files_name:
        file_path = os.path.join(stop_dir, txt_file)
        with open(file_path, 'r', encoding='utf8') as file:
            lines = [read_in_line.rstrip() for read_in_line in file] #rstrip()刪除換行符號    
            # 使用dict.update()方法可以自動覆蓋stopwords中已經讀取存在的停用詞，藉此篩選多檔案中的重複元素。
            stopwords.update({}.fromkeys(lines))  # 利用fromkeys，將內容存進stopwords的key，value set為None。
    return stopwords

def get_words_freq (df,str_var,stop_dir='./source/stopwords'):
    # 請將所有要使用的stopwords檔案儲存在同一個資料夾stop_dir
    # tokenize
    tokens = words_tokenization (df,str_var)
    # Load stopwords
    stop_dir = stop_dir
    stopwords = get_stopwords(stop_dir)
    # 篩去stopwords，計數分詞詞頻加入hash中
    myHash = {};
    for word in tokens: 
        if word not in stopwords:
          if word in myHash:
            myHash[word] += 1 #舊字彙命值+1
          else:		
            myHash[word] = 1 # 新字彙加進myHash命值1
    # 依詞頻由高至低排序存入分詞
    words_freq = [(v, k) for k, v in myHash.items()] # 為了利用list.sort方法由高至低排列詞頻，逐項將key:value轉成(value, key) tuple
    words_freq.sort()
    words_freq.reverse()
    words_freq = [(k, v) for v, k in words_freq] # 成功sort之後將[(value, key)]轉回[(key, value)]
    # 大量資料時dict在Variable Explorer讀取會出問題，list正常，所以維持list輸出。
    return words_freq

def get_words_pop (df_first_dat, str_var, pop_var, stop_dir='./source/stopwords', head_num=300):
    # 請將所有要使用的stopwords檔案儲存在同一個資料夾stop_dir
    print('執行這行，你可以站起來走一走，沖杯咖啡再坐下來品幾口，然後開個yt')
    # **預設只取前三百名的發燒影片的字庫下來給token比對，如果你把全影片拿下去比對，程式跑一次就要喝N杯咖啡才夠做一張圖。**
    df_first_dat = df_first_dat.sort_values(by = pop_var,ascending=False)
    df_first_dat = df_first_dat.head(head_num)
    # tokenize
    tokens = words_tokenization (df_first_dat,str_var)
    # Load stopwords
    stopwords = get_stopwords(stop_dir)
    # 篩去stopwords將pop_var累加入hash中
    print('地獄般的文字比對迴圈開始...')
    myHash = {};
    for word in tokens: 
        if word not in stopwords:
          if word in myHash: # 若字彙已存過pop_var
              for index, row in df_first_dat.iterrows():
                  if word in row[str_var]:
                      myHash[word] += int(row[pop_var]) # 累加pop_var
                      break
          else: # 若是新字彙
              for index, row in df_first_dat.iterrows():
                  if word in row[str_var]:
                      myHash[word] = int(row[pop_var]) # 存入pop_var
                      break
    # 依詞頻由高至低排序存入分詞
    print('迴圈結束了，恭喜你~')
    words_pop = [(v, k) for k, v in myHash.items()] # 為了利用list.sort方法由高至低排列詞頻，逐項將key:value轉成(value, key) tuple
    words_pop.sort()
    words_pop.reverse()
    words_pop = [(k, v) for v, k in words_pop] # 成功sort之後將[(value, key)]轉回[(key, value)]
    # 大量資料時dict在Variable Explorer讀取會出問題，list正常，所以維持list輸出。
    return words_pop

def generate_wordcloud_plot (df_words_weight, weight_var, font_file='msjh.ttc'):
    dict_words_weight = dict(df_words_weight) # 因 wc.generate_from_frequencies 函式所須，轉回dict物件
    # 指定字體檔的路徑
    font_path = r'C:/Users/user/AppData/Local/Microsoft/Windows/Fonts/%s'%font_file
    # # 創建一個Pillow字體物件
    # font = ImageFont.truetype(font_path, size=12)
    # 創建WordCloud物件並設置字體
    wc = WordCloud(font_path=font_path, background_color="black", max_words=1000, width=800, height=600)
    word_cloud = wc.generate_from_frequencies(dict_words_weight[weight_var])
    plt.figure(figsize=(10, 8))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')  
    plt.show()

def bar_range(dict_data):
    bar_range = (min(dict_data.values())*0.7,max(dict_data.values())*1.15)
    return bar_range

def plot_h_barchart_wordpop(df_data, weight_var, head=10, font_family='Microsoft JhengHei', ascending=True, xlim_range=None, xlabel='Frequency', xlabel_size=16, ylabel_size=16):
    df_data = df_data.sort_values(by = weight_var, ascending=False)
    df_data = df_data.head(head)
    keys = list(df_data.index)  # dict指定物件會發生如何翻轉順序，都會重置順序的問題，於是將keys和values分別存成list，處理升冪降冪問題
    values = list(df_data[weight_var])
    
    if not xlim_range:
        xlim_range = (min(values)*0.7, max(values)*1.2)  # 設定 x 軸範圍，允許稍微超過最大值
    else:
        xlim_range = xlim_range

    # 創建深綠到淺綠的漸變色
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_green", ['#004d00', '#99ff99'], N=head)
    colors = [cmap(i / (head - 1)) for i in range(head)]
    
    # 開始作圖
    with plt.style.context(['science', 'no-latex']):
        plt.figure(figsize=(10, 8))
        createplot = plt.subplot()  # 創建新圖表   
        bars = createplot.barh(range(1, head+1), values, align='center', color=colors)  # 繪製基本 barchart
        
        if ascending:  # list_data預設是降冪排列，ascending為真時確保barh由下至上升冪排列
            plt.gca().invert_yaxis()
        
        # 設定x軸範圍
        plt.xlim(xlim_range)
        
        # 加入annotation (含處理emoji)
        for label, y, x in zip(keys, range(1, head+1), values):
            if emoji.is_emoji(label):
                label = emoji.emojize(label)  # 使用 emoji 模組轉換表情符號
                prop = font_manager.FontProperties(family='Segoe UI Emoji')  # Windows 可以用 'Segoe UI Emoji'
            else:
                prop = font_manager.FontProperties(family=font_family)  # 預設文字使用正黑體
            
            plt.annotate(
                label,
                xy=(x, y), xytext=(7, 0),
                textcoords='offset points', ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.5', alpha=0),
                fontproperties=prop,  # 使用支援表情符號的字體
                fontsize=20
            )
        
        createplot.set_yticks([])  # 移除 y 軸標籤
        
        # 設定背景顏色
        plt.gcf().set_facecolor('#f5f5ef')  # 設定背景顏色
        plt.xlabel(xlabel, fontsize=xlabel_size)  # 添加 x 軸標籤
        plt.ylabel('Words', fontsize=ylabel_size)  # 添加 y 軸標籤
    
    plt.show()
    
def plot_ACF_PACF(series,plot_color='#72a955'):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    plot_acf(series, lags=40, ax=axes[0], color=plot_color)
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=16)
    
    plot_pacf(series, lags=40, ax=axes[1], color=plot_color) 
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=16)
  # Expand the y-axis range by 0.1 for both plots
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 0.15, ymax + 0.15)  # Expanding the y-axis by 0.15 on both ends
    ax.set_xlabel('Lags', fontsize=12) # 兩圖x軸相同Lags標籤放在第二張圖就好
        
  # Hide the right, left, and top spines for both plots
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
    # 設置背景顏色
    # ax.set_facecolor('#f5f5ef') # 原色不好看
    
    plt.tight_layout()
    plt.show()
    
def ARIMA_model_fit(series, pdq=(7, 1, 0)):
    model = ARIMA(series, order=pdq)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def find_optimal_split_x(df_data, start_x_index, end_x_index, dependent_variable, independent_variables,model_family=sm.families.Poisson()):
    # 跑完log-likelihood的標準差只有約9.9e-10，要繪圖或運算需要放大處理
    best_likelihood = -np.inf
    best_x = None
    assumed_model = f'{dependent_variable} ~ {independent_variables}'
    # 用df_index取得x day迭帶的起始與終止日
    start_date = df_data['DoDiag'].loc[start_x_index]
    start_date = start_date.to_pydatetime()
    start_date = start_date.replace(tzinfo=None)
    end_date = df_data['DoDiag'].loc[end_x_index]
    end_date = end_date.to_pydatetime()
    end_date = end_date.replace(tzinfo=None)
    # 取得全資料頭尾日期
    min_date, max_date = get_min_date_max_date(df_data['DoDiag'])
    df_track = pd.DataFrame({'date_x':[],'log-likelihood':[]})
    print('開始依據x day逐一計算log-likelihood...')
    for x in pd.date_range(start=start_date, end=end_date):
        # 分割數據(before_x:頭日至x-1日；after_x:x日至尾日)
        x = x.to_pydatetime()
        before_x = df_data[df_data['DoDiag'].between(min_date, x + relativedelta(days= -1))]
        after_x = df_data[df_data['DoDiag'].between(x, max_date)]
        
        # glm Poisson模型
        model_before = smf.glm(formula = assumed_model, data=before_x, family=model_family).fit()
        model_after = smf.glm(formula = assumed_model, data=after_x, family=model_family).fit()
        # likelihood相加
        likelihood = model_before.llf + model_after.llf
        
        df_track = df_track.append({'date_x': x, 'log-likelihood': likelihood}, ignore_index=True)
        
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_x = x

    return best_x, df_track, start_date, end_date

def plot_series_data_compare(Series1,Series2):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(Series1)
    plt.subplot(1, 2, 2)
    plt.plot(Series2)
    plt.show()

def plot_norm_hist_qq(Series):
    # 繪製直方圖
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    # sns.histplot(Series, kde=True, bins = 'auto')
    plt.hist(Series, bins=20, density=True, alpha=0.6, color='g')
    plt.xlabel('YJ_NewCase')
    plt.ylabel('frequency')
    plt.title('YJ_NewCase histogram')
    plt.grid(True)

    # 繪製 Q-Q 圖
    plt.subplot(1, 2, 2)
    stats.probplot(Series, dist="norm", plot=plt)
    plt.title('YJ_NewCase Q-Q Plot')

    plt.tight_layout()
    plt.show()
 
#%% set workspace
# worksp_path = r"D:\tmp\pythonProject\pythonProject"
set_workspace() # 預設利用檔案當前路徑，改變路徑請設定worksp_path帶入
print("Current WORKSPACE: "+os.getcwd())
#%% load data
dat_yt, df_dup_yt = load_data_yt() # dat_yt:篩選好的資料； df_dup_yt: 留下哪些資料是異常重複的
dat_tot_cov19 = load_data_covid()
dat_cov19 = covid_between_yt_period(dat_tot_cov19,dat_yt) # yt到2023年12月底，但cov19只到2023/9
# print(dat_cov.dtypes," ",dat_yt.dtypes)
# load df_words_pop 請到 wordcloud章節中後段執行
#%% plot line chart of dat_cov19    (建立dat_cov19_4plot副本)
dat_cov19_4plot = dat_cov19.copy()
dat_cov19_4plot.set_index('DoDiag', inplace=True)

upper_bound = dat_cov19_4plot['7ma_NwCase'] + dat_cov19_4plot['7mstd_NwCase']  # 設置上界
lower_bound = dat_cov19_4plot['7ma_NwCase'] - dat_cov19_4plot['7mstd_NwCase']  # 設置下界

fig, ax = plt.subplots(figsize=(12, 7))
plot_rc()

ax.plot(dat_cov19_4plot.index, dat_cov19_4plot['NwCase'], label='Daily confirmed Cases',
         color='#374a48',
         linewidth=1.5
         )
ax.fill_between(dat_cov19_4plot.index, lower_bound, upper_bound,
                 color='#cacac5', alpha=0.6,
                 label='Standard Deviation'
                 )

ax.plot(dat_cov19_4plot.index, dat_cov19_4plot['7ma_NwCase'], label='7days-moving average',
         color='#72a955',
         linestyle="--",
         linewidth=2
         ) 
ax.legend()
# 設置背景顏色
ax.set_facecolor('#f5f5ef')

# Hide the all but the bottom spines (axis lines)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)

fig.savefig('Line_chart_cov19.png')

#%% Gantt Chart for dat_cov19 & dat_yt
#=========================
# dataset
#=========================
min_date_yt, max_date_yt = get_min_date_max_date(dat_yt['trending_time'])
prd_yt = max_date_yt - min_date_yt
prd_yt = prd_yt.days
min_date_cov19, max_date_cov19 = get_min_date_max_date(dat_tot_cov19['DoDiag'])
prd_cov19 = max_date_cov19 - min_date_cov19
prd_cov19 = prd_cov19.days
prd_cov19_chunk = max_date_cov19 - min_date_yt
prd_cov19_chunk = prd_cov19_chunk.days

chart_start_date = min_date_cov19 + relativedelta(months= -3) # 用delta設定圖的x時間軸顯示範圍
chart_end_date = max_date_yt + relativedelta(months= 3)

df_Gantt = pd.DataFrame({'Datasets':['YouTube_trend','COVID19_confirmation'],
                         'start_date':[min_date_yt, min_date_cov19], # 半透chart用
                         'chunk_date':[min_date_yt, min_date_yt], # 實心chart用
                         'Periods':[prd_yt,prd_cov19], # 半透chart用
                         'Chunk_periods':[prd_yt,prd_cov19_chunk] # 實心chart用
                         })
#=========================
# generate plot
#=========================
fig, ax = plt.subplots(1, figsize=(12, 5))
plot_rc()
chart_color = '#72a955'
y_positions = [0.6, 1]

# 資料集全距 (半透明)
ax.barh(y=y_positions, width=df_Gantt.Periods,left=df_Gantt.start_date, color=chart_color, alpha=0.5, height=0.2)
# 擷取範圍 (實心)
ax.barh(y=y_positions, width=df_Gantt.Chunk_periods,left=df_Gantt.chunk_date, color=chart_color, alpha=1.0, height=0.2)

# 設定 x 軸日期範圍和標籤
ax.set_xlim(pd.Timestamp(chart_start_date), pd.Timestamp(chart_end_date))

# 添加每個月份的垂直格線 (在長條後面)
ax.grid(axis='x', color='white', linestyle='--', linewidth=0.5, zorder=0, alpha=0.3)
# 去除y軸單位標籤
ax.tick_params(left=False, labelleft=False)

# 設置背景顏色
ax.set_facecolor('#304543')

# 在每個 chart 新增標籤
for idx, row in df_Gantt.iterrows():
    ax.text(row['start_date'] + pd.Timedelta(days=row['Periods'] / 2),
            y_positions[idx], row['Datasets'], va='center', ha='center', color='white', fontweight='bold')
ax.set_title("Time Line")

# fig.savefig('Gantt_Chart_Timeline.png')

#%% *Define the end point of the epidemic*    (建立dat_cov19_4modelfit副本)
'''
ACF與PACF顯示NwCase有平滑(拖尾)自相關性，特別在前三項與潛在第七項(yj變化後7th偏自相關消失，時間關係先無法深入探討)
原先希望透過ARIMA model擬合NwCase(每日新增確診人數)計算趨勢與評估移動平均為零之假說檢定，
但因發現Yeo-Johnson轉換後資料不但沒轉成近似常態，反而變成U型谷，QQ-plot成Z字型而作罷。 
plot Yeo-Johnson轉換NwCase發現資料可視作明確的two-phase trasition，因此形成新的模型假設如下:
存在一未知變量，使疫情靜息日(暫稱 x-day) 開始，新增確診率大幅降低，
因此疫情期間獨立確診率服從特定機率 lambda_before_x，模型假設 model_1 如下:
Cas_i ~ Cas_i-1 + Cas_i-2 + ...(自相關項) + DoDiag_ym (月份常數) + error
error ~ Poi(lambda_before_x)
疫情經過x-day開始靜息，服從另一個獨立確診率lambda_after_x，模型假設 model_2 如下:
Cas_i ~ Cas_i-1 + Cas_i-2 + ...(自相關項) + DoDiag_ym (月份常數) + error (與上式相同)
error ~ Poi(lambda_after_x)
%疫情斷點定義方法:
    利用逐一嘗試設定x-day為任一日，計算model_1與 model_2在GLM(Poi)擬合之log-likelihood做相加，
    取得log-likelihood和最大值，定義為最佳x-day估計值。
%自回歸項數篩選方法:
    對dat_cov19['NwCase']進行GLM(Poi)擬合，取得擬合BIC值。
    並且逐一增加、剔除自相關項與月份常數，放到疫情斷點定義方法取得最佳x-day時的model_1與model_2之BIC值加總。
    綜合評估全資料BIC與斷點資料BIC和，判斷當中的最適模型。
'''
# 建立dat_cov19_4modelfit副本，避免分析資料汙染原始資料
dat_cov19_4modelfit = dat_cov19.copy()
# NwCase進行Yeo-Johnson轉換存入YJtrans_NwCase (診斷數存在0值，不能使用Box-Cox轉換)
YJtrans_NwCase_cov19, best_lambda = stats.yeojohnson(dat_cov19_4modelfit['NwCase'])
dat_cov19_4modelfit.loc[:, 'YJtrans_NwCase'] = pd.Series(YJtrans_NwCase_cov19).values
# tot_YJtrans_NwCase_cov19, best_lambda = stats.yeojohnson(dat_tot_cov19['NwCase']) # 現在用不太到(依yt時間分段前之全資料NwCaseyj轉換)

# 繪製 ACF 和 PACF 圖
plot_ACF_PACF(dat_cov19_4modelfit['NwCase'])
# plot_ACF_PACF(YJtrans_NwCase_cov19) # 現在用不太到
# plot_ACF_PACF(tot_YJtrans_NwCase_cov19)

# # 繪製hist與QQ plot (轉換模型假設後，現在用不太到)
# plot_norm_hist_qq(YJtrans_NwCase_cov19)
# plot_norm_hist_qq(YJtrans_NwCase_cov19[:104])
# plot_norm_hist_qq(YJtrans_NwCase_cov19[106:])

# 全數據擬合
model = smf.glm(formula='NwCase ~ DoDiag_ym+delay1_NwCase+delay2_NwCase+delay3_NwCase', data=dat_cov19_4modelfit, family=sm.families.Poisson()).fit()
print(model.bic)
print(model.summary())
'''
範圍全數據利用bic進行模型篩選:
    NwCase ~ DoDiag_ym+delay1_NwCase+delay2_NwCase+delay3_NwCase+delay4_NwCase: 182255.98 (min)
    NwCase ~ DoDiag_ym+delay1_NwCase+delay2_NwCase+delay3_NwCase: 183295.78
    NwCase ~ DoDiag_ym+delay1_NwCase+delay2_NwCase: 187446.49
    NwCase ~ DoDiag_ym+delay1_NwCase: 188100.86
    NwCase ~ DoDiag_ym: 287240.08
    NwCase ~ delay1_NwCase: 957579.67 (max)
    NwCase ~ delay1_NwCase+delay2_NwCase: 917859.30
    NwCase ~ delay1_NwCase+delay2_NwCase+delay3_NwCase: 848655.02
    NwCase ~ delay1_NwCase+delay2_NwCase+delay3_NwCase+delay4_NwCase: 819551.32
結果顯示 DoDiag_ym+delay1_NwCase+delay2_NwCase+delay3_NwCase+delay4_NwCase 是當中最適合的模型
    
'''

# 全數據擬合的殘差圖
plt.figure(figsize=(8, 6))
residuals = model.resid_response
plt.plot(residuals)
plt.show()

# 找尋最佳靜息日(x-day)
dep_var='NwCase'
ind_var='DoDiag_ym+delay1_NwCase+delay2_NwCase+delay3_NwCase'
# 用數字index搜尋x-day
start_x_index = 50
end_x_index = 210
optimal_x, likelihood_track, star_searchx, end_searchx = find_optimal_split_x(dat_cov19_4modelfit,
                                                                              start_x_index=start_x_index, end_x_index=end_x_index,
                                                                              dependent_variable=dep_var,independent_variables=ind_var
                                                                              )
print("最佳 X day:", optimal_x)
# 檢視最佳分斷時的模型擬合特性
# 用最佳x-day回頭建立資料分段:x前dat_cov19:before_x； x後dat_cov19:after_x
min_date, max_date = get_min_date_max_date(dat_cov19_4modelfit['DoDiag'])
before_x = dat_cov19_4modelfit[dat_cov19_4modelfit['DoDiag'].between(min_date, optimal_x + relativedelta(days= -1))]
after_x = dat_cov19_4modelfit[dat_cov19_4modelfit['DoDiag'].between(optimal_x, max_date)]

# glm Poisson模型
model_before_x = smf.glm(formula=f'{dep_var} ~ {ind_var}', data=before_x, family=sm.families.Poisson()).fit()
model_after_x = smf.glm(formula=f'{dep_var} ~ {ind_var}', data=after_x, family=sm.families.Poisson()).fit()
print(model_before_x.bic + model_after_x.bic)
print(model_before_x.summary())
print(model_after_x.summary())

'''
optimal_x截斷數據bic相加結果骰選模型結果如下:
    NwCase ~ DoDiag_ym+delay1_NwCase+delay2_NwCase+delay3_NwCase+delay4_NwCase: 65585.4 (min) #有數字算到值域外? ValueError: NaN, inf or invalid value detected in weights, estimation infeasible.
    NwCase ~ DoDiag_ym+delay1_NwCase+delay2_NwCase+delay3_NwCase: 66386.82
    NwCase ~ DoDiag_ym+delay1_NwCase+delay2_NwCase: 67974.32
    NwCase ~ DoDiag_ym+delay1_NwCase: 69896.10
    NwCase ~ DoDiag_ym: 130564.31 (max)
    NwCase ~ delay1_NwCase: 91554.59
    NwCase ~ delay1_NwCase+delay2_NwCase: 87867.72
    NwCase ~ delay1_NwCase+delay2_NwCas+delay3_NwCase: 83823.07
結果顯示 DoDiag_ym+delay1_NwCase+delay2_NwCase+delay3_NwCase 是當中最適合的模型
'''
# model_1 & model_2殘差圖
residuals_b = model_before_x.resid_response
residuals_a = model_after_x.resid_response
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(residuals_b)
plt.subplot(2, 1, 2)
plt.plot(residuals_a)
plt.show()

# 繪製預測線與原始數據
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(before_x['NwCase'])
plt.plot(model_before_x.predict())
plt.subplot(2, 1, 2)
plt.plot(after_x['NwCase'].reset_index(drop=True))
plt.plot(model_after_x.predict())
plt.show()

#%% plot NwCase & YJ tramsformed NwCase    (仰賴dat_cov19_4plot)
YJtrans_NwCase_cov19, best_lambda = stats.yeojohnson(dat_cov19_4plot['NwCase'])
dat_cov19_4plot.loc[:, 'YJtrans_NwCase'] = pd.Series(YJtrans_NwCase_cov19).values

# fig, ax = plt.subplots(1, 2, 1, figsize=(12, 8))
plt.figure(figsize=(12, 8))
plot_rc()
plt.subplot(2, 1, 1)
plt.plot(dat_cov19_4plot.index, dat_cov19_4plot['NwCase'], label='Daily confirmed Cases',
         color='#374a48',
         linewidth=1.5
         )
plt.title('National COVID-19 daily confirmed statistics chart')

plt.subplot(2, 1, 2)
plt.plot(dat_cov19_4plot.index, dat_cov19_4plot['YJtrans_NwCase'], label='YJ tramsformed daily confirmed Cases',
         color='#374a48',
         linewidth=1.5
         )
plt.title('Yeo-Johnson trasformed daily confirmed Cases')

plt.tight_layout()
plt.show()

#%% plot NwCase & sum of likelihood    (仰賴dat_cov19_4plot，建立dat_cov19_4likeplot)
# 建立dat_cov19_4likeplot
dat_cov19_4likeplot = dat_cov19_4modelfit.copy()
# 取時間範圍資料與likelihood track資料等長
dat_cov19_4likeplot = dat_cov19_4likeplot[dat_cov19_4likeplot['DoDiag'].between(star_searchx, end_searchx)]
dat_cov19_4likeplot.set_index('DoDiag', inplace=True)

highlight_index = dat_cov19_4likeplot.index.get_loc(optimal_x)  # 找到對應的 index 位置

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
plot_rc()

# 第一張圖
axes[0].plot(dat_cov19_4likeplot.index, dat_cov19_4likeplot['NwCase'], label='Daily confirmed Cases',
             color='#374a48', linewidth=1.5)

# 標註圓點
axes[0].scatter(dat_cov19_4likeplot.index[highlight_index], dat_cov19_4likeplot['NwCase'].iloc[highlight_index],
                color='red', zorder=5, s=50)

# 標註日期
axes[0].text(dat_cov19_4likeplot.index[highlight_index], dat_cov19_4likeplot['NwCase'].iloc[highlight_index],
             'Optimal x-day: 2023-03-19', color='red', verticalalignment='bottom')
# 消去 x 軸的刻度線和標籤
axes[0].tick_params(axis='x', which='both', labelbottom=False)

axes[0].set_title('National COVID-19 daily confirmed statistics chart')

# 設置背景顏色
axes[0].set_facecolor('#f5f5ef')

# 第二張圖
axes[1].plot(dat_cov19_4likeplot.index, likelihood_track['log-likelihood'], label='Sum of Log-likelihood',
             color='#374a48', linewidth=1.5)

# 標註圓點
axes[1].scatter(dat_cov19_4likeplot.index[highlight_index], likelihood_track['log-likelihood'].iloc[highlight_index],
                color='red', zorder=5, s=50)

# 標註日期
axes[1].text(dat_cov19_4likeplot.index[highlight_index], likelihood_track['log-likelihood'].iloc[highlight_index],
             'Optimal x-day: 2023-03-19', color='red', verticalalignment='bottom')

axes[1].set_title('Sum of Log-likelihood')

# 設置背景顏色
axes[1].set_facecolor('#f5f5ef')

# 調整 y 軸範圍，根據你的 ax 修改成對應的 axes[1] 設定
ymin, ymax = axes[1].get_ylim()
axes[1].set_ylim(ymin, ymax + 2000)

# Hide the right, left, and top spines for both plots
for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

#%% wordcloud: afterx beforex (仰賴optimal_x,建立dat_yt_last_video， dat_yt_first_video 副本)
# generate df words' frequency & popularity data
# 建立dat_yt_last_video
dat_yt_last_video = dat_yt.copy() # 建立資料副本，避免指向同一物件
dat_yt_last_video = dat_yt_last_video.sort_values(by = 'trending_time',ascending=False)
dat_yt_last_video.drop_duplicates(subset='video_id',keep='last', inplace=True) # 將好幾期持續發燒的影片只留下最後一筆資料(依照 video_id 篩選)
# 建立dat_yt_first_video (發燒時間定義: 初次上發燒； popularity定義: 最後一次上發燒的累積資料)
dat_yt_first_video = dat_yt.copy() # 建立資料副本，避免指向同一物件
dat_yt_first_video = dat_yt_last_video.sort_values(by = 'trending_time',ascending=False)
dat_yt_first_video.drop_duplicates(subset='video_id',keep='first', inplace=True) # 將好幾期持續發燒的影片留下第一筆資料(依照 video_id 篩選)
dat_yt_first_video['last_trend_time'] = dat_yt_last_video['trending_time']
dat_yt_first_video['last_trend_view'] = dat_yt_last_video['view']
# ===================================
# 根據x-day做資料(dat_yt_first_video)分斷
# ===================================
min_date, max_date = get_min_date_max_date(dat_yt_first_video['trending_time'])
# 現版本的pd.Series datetime64[ns, UTC] 轉換為 datetime.datetime看起來極難解決 (寫這個架構的4 hole familymart)
# 將min_date, max_date與optimal_x 轉換為 datetime64[ns, UTC]
min_date = pd.to_datetime(min_date).tz_localize('UTC')
max_date = pd.to_datetime(max_date).tz_localize('UTC')
optimal_x64 = pd.to_datetime(optimal_x).tz_localize('UTC')
# 根據x-day做資料(dat_yt_first_video)分斷
dat_yt_before_x = dat_yt_first_video[dat_yt_first_video['trending_time'].between(min_date, optimal_x64 + relativedelta(days= -1))]
dat_yt_after_x = dat_yt_first_video[dat_yt_first_video['trending_time'].between(optimal_x64, max_date)]
# ================================================
# get token list counted by word frequency
# ================================================
str_var = 'title' # 要分析的字庫群:'title'或'description'(或'channel name'、'tags'?)
# stop_dir = './source/stopwords' # 請將所有要使用的stopwords檔案儲存在同一個資料夾stop_dir
words_freq = get_words_freq(dat_yt_last_video,str_var)
words_freq_before_x = get_words_freq(dat_yt_before_x,str_var)
words_freq_after_x = get_words_freq(dat_yt_after_x,str_var)

df_words_freq = pd.DataFrame(words_freq, columns=['words', 'frequency'])
df_words_freq.set_index('words', inplace=True)
df_words_freq_before_x = pd.DataFrame(words_freq_before_x, columns=['words', 'frequency'])
df_words_freq_before_x.set_index('words', inplace=True)
df_words_freq_after_x = pd.DataFrame(words_freq_after_x, columns=['words', 'frequency'])
df_words_freq_after_x.set_index('words', inplace=True)

'''
******** Warning: get_words_pop比對工作量龐大，以下code可以打開測試迴圈正常運作 ********

全影片words_pop, words_pop_before_x, words_pop_after_x三筆資料實測約兩個半小時跑完。
words_pop比對數19814*4746=94037244(九千四百多萬次比對)
words_pop_before_x比對數7811*4746=37071006(三千七百多萬次比對)
words_pop_after_x比對數15307*4746=72647022(七千六百多萬次比對)
生圖分析先一次痛苦，把全資料跑出來存成csv和json，之後直接取用事前儲存的words_pop資料集來用。
以下已將參數設定為只取前50名觀看數影片來抓字彙(自訂函數時預設是取前300名影片)，可以放心測試。
'''
# # token list counted by the word popularity variable (一筆要跑很久^^) (預設只取前300名影片)
# pop_var = 'last_trend_view'
# words_pop = get_words_pop(dat_yt_first_video,str_var, pop_var, head_num=50) 
# words_pop_before_x = get_words_pop(dat_yt_before_x,str_var, pop_var, head_num=50) 
# words_pop_after_x = get_words_pop(dat_yt_after_x,str_var, pop_var, head_num=50)
# # 儲存全影片字庫資料 (csv, json)
# df_words_pop = pd.DataFrame(words_pop, columns=['words', 'tot_view'])
# df_words_pop.to_csv('words_pop.csv', encoding = 'utf-8-sig') 
# df_words_pop.set_index('words', inplace=True)
# df_words_pop.to_json('words_pop.json') 
# # 儲存before_x影片字庫資料 (csv, json)
# df_words_pop_before_x = pd.DataFrame(words_pop_before_x, columns=['words', 'tot_view'])
# df_words_pop_before_x.to_csv('words_pop_before_x.csv', encoding = 'utf-8-sig') 
# df_words_pop_before_x.set_index('words', inplace=True)
# df_words_pop_before_x.to_json('words_pop_before_x.json') 
# # 儲存after_x影片字庫資料 (csv, json)
# df_words_pop_after_x = pd.DataFrame(words_pop_after_x, columns=['words', 'tot_view'])
# df_words_pop_after_x.to_csv('words_pop_after_x.csv', encoding = 'utf-8-sig') 
# df_words_pop_after_x.set_index('words', inplace=True)
# df_words_pop_after_x.to_json('words_pop_after_x.json') 
# ================================================
# load df_words_pop
# ================================================
df_words_pop = pd.read_json("./Export/words_pop/words_pop.json")
df_words_pop_before_x = pd.read_json("./Export/words_pop/words_pop_before_x.json")
df_words_pop_after_x = pd.read_json("./Export/words_pop/words_pop_after_x.json")

df_words_pop = df_words_pop.sort_values(by = 'tot_view',ascending=False)
df_words_pop_before_x = df_words_pop_before_x.sort_values(by = 'tot_view',ascending=False)
df_words_pop_after_x = df_words_pop_after_x.sort_values(by = 'tot_view',ascending=False)
# ================================================
# plot wordcloud & hchart
# ================================================
# generate plots
font_file = '瀞ノグリッチ黒体H2.otf'
generate_wordcloud_plot(df_words_freq, weight_var='frequency',font_file=font_file)
plot_h_barchart_wordpop(df_words_freq, weight_var='frequency')

generate_wordcloud_plot(df_words_freq_before_x, weight_var='frequency')
plot_h_barchart_wordpop(df_words_freq_before_x, weight_var='frequency')

generate_wordcloud_plot(df_words_freq_after_x, weight_var='frequency')
plot_h_barchart_wordpop(df_words_freq_after_x, weight_var='frequency')

# words_pop wordcloud (View)
generate_wordcloud_plot(df_words_pop, weight_var='tot_view',font_file=font_file)
plot_h_barchart_wordpop(df_words_pop, weight_var='tot_view', xlabel='total_view')

generate_wordcloud_plot(df_words_pop_before_x, weight_var='tot_view')
plot_h_barchart_wordpop(df_words_pop_before_x, weight_var='tot_view', xlabel='total_view')

generate_wordcloud_plot(df_words_pop_after_x, weight_var='tot_view')
plot_h_barchart_wordpop(df_words_pop_after_x, weight_var='tot_view', xlabel='total_view')
# =====================================
# df_words_pop (view) X前後 做YJ轉換繪製h_barchart
# =====================================
totdat_YJtrans_wordview_before_x, best_lambda = stats.yeojohnson(df_words_pop_before_x['tot_view'])
totdat_YJtrans_wordview_after_x, best_lambda = stats.yeojohnson(df_words_pop_after_x['tot_view'])

df_words_pop_before_x.loc[:, 'totdat_YJwordview_before_x'] = pd.Series(totdat_YJtrans_wordview_before_x).values
df_words_pop_after_x.loc[:, 'totdat_YJwordview_after_x'] = pd.Series(totdat_YJtrans_wordview_after_x).values

plot_h_barchart_wordpop(df_words_pop_before_x, weight_var='totdat_YJwordview_before_x',head=25, xlabel='total_view_before_x (YJ transformed)')
plot_h_barchart_wordpop(df_words_pop_after_x, weight_var='totdat_YJwordview_after_x',head=25, xlabel='total_view_after_x (YJ transformed)')

#%% Paired t-test of ranked word frequency (數據型態不理想)
# 將words_freq_before_x與words_freq_after_x轉換成df
df_words_freq_before_x = pd.DataFrame(words_freq_before_x, columns=['words', 'frequency'])
df_words_freq_after_x = pd.DataFrame(words_freq_after_x, columns=['words', 'frequency'])
# 利用pd.merge將兩個df取交集，存成df_inter_word_freq
df_inter_word_freq = pd.merge(df_words_freq_before_x, df_words_freq_after_x, on='words', how='inner', suffixes=('_before_x', '_after_x'))

# 將frequency分別轉換成 rank
df_inter_word_freq['rank_before_x'] = df_inter_word_freq['frequency_before_x'].rank(ascending=False, method='dense') # method='dense'代表同列名次之後不會有空隙名次
df_inter_word_freq['rank_after_x'] = df_inter_word_freq['frequency_after_x'].rank(ascending=False, method='dense')

# 將rank相減並做paired t-test
df_inter_word_freq['rank_diff'] = df_inter_word_freq['rank_before_x']-df_inter_word_freq['rank_after_x']
# =====================================
# 使用OLS模型針對word frequency rank 做t-test 
# =====================================
# word frequency的中後段rank冪次太髒(同frequnecy並列名次數據太多)使t-test結果估計約為-24.4偏掉的95%CI失去意義

X_ttest = np.ones(len(df_inter_word_freq))  # 以1序列作為常數變數 (Intercept)
y_ttest = df_inter_word_freq['rank_diff']

model_t = sm.OLS(y_ttest, X_ttest).fit()  # Ordinary Least Squares 模型
print(model_t.summary()) # 檢定結果失去意義

#%% Paired t-test of word view
# =====================================
# 產生df_inter_word_view (兩圖交集) 先做YJ轉換再做Z-score標準化
# =====================================
# 利用pd.merge將兩個df取交集，存成df_inter_word_freq
df_inter_word_view = pd.merge(df_words_pop_before_x, df_words_pop_after_x, left_index=True, right_index=True,  how='inner', suffixes=('_before_x', '_after_x'))
YJtrans_wordview_before_x, best_lambda = stats.yeojohnson(df_inter_word_view['tot_view_before_x'])
YJtrans_wordview_after_x, best_lambda = stats.yeojohnson(df_inter_word_view['tot_view_after_x'])

df_inter_word_view.loc[:, 'YJwordview_before_x'] = pd.Series(YJtrans_wordview_before_x).values
df_inter_word_view.loc[:, 'YJwordview_after_x'] = pd.Series(YJtrans_wordview_after_x).values

# 創建scaler物件，可以對物件進行z轉換(標準化)
scaler = StandardScaler()
YJtrans_wordview_before_x_reshaped = YJtrans_wordview_before_x.reshape(-1, 1) # 要用reshape方法建立第二維(即便是1)，才能做fit_transform
sta_YJview_before_x = scaler.fit_transform(YJtrans_wordview_before_x_reshaped)

YJtrans_wordview_after_x_reshaped = YJtrans_wordview_after_x.reshape(-1, 1) # 要用reshape方法建立第二維(即便是1)，才能做fit_transform
sta_YJview_after_x = scaler.fit_transform(YJtrans_wordview_after_x_reshaped)

sta_YJview_before_x = sta_YJview_before_x.reshape(-1) # 轉回1D array才能直接塞給Series
sta_YJview_after_x = sta_YJview_after_x.reshape(-1)
df_inter_word_view.loc[:, 'sta_YJview_before_x'] = pd.Series(sta_YJview_before_x).values
df_inter_word_view.loc[:, 'sta_YJview_after_x'] = pd.Series(sta_YJview_after_x).values
# =====================================
# 計算差值利用OLS建立t分配模型
# =====================================
df_inter_word_view['diff_sta_YJ_view'] = df_inter_word_view['sta_YJview_before_x']-df_inter_word_view['sta_YJview_after_x']

X_ttest = np.ones(len(df_inter_word_view))  # 以1序列作為常數變數 (Intercept)
y_ttest = df_inter_word_view['diff_sta_YJ_view']

model_t = sm.OLS(y_ttest, X_ttest).fit()  # Ordinary Least Squares 模型
print(model_t.summary()) # p-value是完美的1.0，可以拿來做分類

# =====================================
# 利用OLSt分配模型進行文字分群:
#    group1:疫情前後不變； group2:疫情後顯著萎縮 group3:疫情後顯著成長
# =====================================
# 取得 OLS 模型的 95% 信賴區間
conf_int = model_t.conf_int(alpha=0.05)
CI_025 = conf_int.iloc[0, 0]  # 2.5% CI
CI_975 = conf_int.iloc[0, 1]  # 97.5% CI

# 進行分群
df_group1_wordview = df_inter_word_view[(df_inter_word_view['diff_sta_YJ_view'] > CI_025) & (df_inter_word_view['diff_sta_YJ_view'] < CI_975)]
df_group2_wordview = df_inter_word_view[df_inter_word_view['diff_sta_YJ_view'] >= CI_975]
df_group3_wordview = df_inter_word_view[df_inter_word_view['diff_sta_YJ_view'] <= CI_025]

#%% plot & Save Group1, 2, 3 (仰賴df_groupN_wordview)
# =============
# Plot
# =============
# Group1
generate_wordcloud_plot(df_group1_wordview, weight_var='YJwordview_after_x')
plot_h_barchart_wordpop(df_group1_wordview, weight_var='tot_view_after_x',head=10, xlabel='View')
plot_h_barchart_wordpop(df_group1_wordview, weight_var='YJwordview_after_x',head=25, xlabel='View_(YJ transformed)')
# Group2
generate_wordcloud_plot(df_group2_wordview, weight_var='YJwordview_before_x')
plot_h_barchart_wordpop(df_group2_wordview, weight_var='YJwordview_before_x',head=25, xlabel='View (before_x, YJ transformed)')
plot_h_barchart_wordpop(df_group2_wordview, weight_var='diff_sta_YJ_view',head=15, xlabel='View_decrease')
# Group3
df_group3_wordview['neg_diff_sta_YJ_view'] = df_group3_wordview['diff_sta_YJ_view']*-1
generate_wordcloud_plot(df_group3_wordview, weight_var='YJwordview_after_x')
plot_h_barchart_wordpop(df_group3_wordview, weight_var='YJwordview_after_x',head=25, xlabel='View_(YJ transformed)')
plot_h_barchart_wordpop(df_group3_wordview, weight_var='neg_diff_sta_YJ_view',head=15, xlabel='View_ascend')
# ============
# CSV
# ============
# df_group1_wordview.to_csv('group1_wordview.csv', encoding = 'utf-8-sig')
# df_group2_wordview.to_csv('group2_wordview.csv', encoding = 'utf-8-sig')
# df_group3_wordview.to_csv('group3_wordview.csv', encoding = 'utf-8-sig')

#%% generate t probability densisty function (仰賴model_t,CI_025,CI_975)

plot_rc()
# 取得模型的自由度
df_resid = model_t.df_resid  # 殘差的自由度
params = model_t.params[0]  # OLS 模型的係數
stderr = model_t.bse[0]  # 標準誤

# 生成 x 軸數據點 (範圍 -0.055 到 0.055)
x_values = np.linspace(-0.055, 0.055, 500)

# 計算 t 分布的 PDF
pdf = t.pdf(x_values, df_resid, loc=params, scale=stderr)

# 取得 95% 信賴區間
conf_int = model_t.conf_int(alpha=0.05)
CI_025 = conf_int.iloc[0, 0]  # 2.5% CI
CI_975 = conf_int.iloc[0, 1]  # 97.5% CI
# 創建圖形並設置背景顏色
plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.set_facecolor('#304543')  # 設置背景顏色為 #304543
# 填充 Group1 區域 (CI_025 和 CI_975 之間)
plt.fill_between(x_values, pdf, where=(x_values > CI_025) & (x_values < CI_975),
                 color='#66b2b2', alpha=0.7)
# 填充 Group2 區域 (CI_975 以上)
plt.fill_between(x_values, pdf, where=(x_values >= CI_975),
                 color='#ff9999', alpha=0.7)
# 填充 Group3 區域 (CI_025 以下)
plt.fill_between(x_values, pdf, where=(x_values <= CI_025),
                 color='#99cc99', alpha=0.7)
# 標示 Group1 標籤
group1_center = (CI_025 + CI_975) / 2
plt.text(group1_center, 0.1, 'Group1', horizontalalignment='center', fontsize=12, color='white')
# 標示 Group2 標籤
group2_center = (CI_975 + max(x_values)) / 2 - 0.005
plt.text(group2_center, 0.1, 'Group2', horizontalalignment='center', fontsize=12, color='white')
# 標示 Group3 標籤
group3_center = (CI_025 + min(x_values)) / 2 + 0.005
plt.text(group3_center, 0.1, 'Group3', horizontalalignment='center', fontsize=12, color='white')
# 設置標籤和標題 (使用白色字體)
plt.title("Student's t-distribution with Confidence Interval and Groups")
plt.xlabel('Difference of standardized total view')
plt.ylabel('Probability Density')
# 隱藏邊框線條 (只保留色塊)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# 顯示圖形
plt.show()

#%% get group4, group5. plot & save
# ============
# group4: 疫情後消失的字彙； group5: 疫情後出現的字彙
# ============
df_group4_wordview = df_words_pop_before_x[~df_words_pop_before_x.index.isin(df_words_pop_after_x.index)]
plot_h_barchart_wordpop(df_group4_wordview, weight_var='tot_view',head=15, xlabel='total_view')

df_group5_wordview = df_words_pop_after_x[~df_words_pop_after_x.index.isin(df_words_pop_before_x.index)]
plot_h_barchart_wordpop(df_group5_wordview, weight_var='tot_view',head=15, xlabel='total_view')

# ============
# CSV
# ============
# df_group4_wordview.to_csv('group4_wordview.csv', encoding = 'utf-8-sig')
# df_group5_wordview.to_csv('group5_wordview.csv', encoding = 'utf-8-sig')

#%% 用venn2劃出集合圖
plot_rc()
# 建立圖形和座標軸
fig, ax = plt.subplots()

# 畫兩個圓圈
v = venn2(subsets=(1, 1, 0.5), set_labels=('', ''))

# 填充左邊圓圈外部部分的顏色（例如紅色）
v.get_label_by_id('10').set_text('')  # 移除左邊圓的數字標籤
v.get_patch_by_id('10').set_color('#66b2b2')  # 設定左邊圓外的顏色
v.get_patch_by_id('10').set_edgecolor('none')  # 移除邊框

# 填充右邊圓圈外部部分的顏色（例如藍色）
v.get_label_by_id('01').set_text('')  # 移除右邊圓的數字標籤
v.get_patch_by_id('01').set_color('#ff9999')  # 設定右邊圓外的顏色
v.get_patch_by_id('01').set_edgecolor('none')  # 移除邊框

# 移除交集部分的顏色
v.get_patch_by_id('11').set_color('none')  # 設定交集部分的顏色為透明
v.get_patch_by_id('11').set_edgecolor('none')  # 移除邊框
v.get_label_by_id('11').set_text('')  # 移除交集部分的數字標籤

# 在圖形中央標註 "group 4" 和 "group 5"
plt.text(-0.45, 0, 'Group 4', horizontalalignment='center', verticalalignment='center')
plt.text(0.45, 0, 'Group 5', horizontalalignment='center', verticalalignment='center')

# 去除 x 軸標籤
ax.set_xticks([])
ax.set_xticklabels([])

# 顯示圖形
plt.show()

# 儲存圖形為透明背景的 PNG
plt.savefig('group4_5_inters_venn.png', transparent=True, bbox_inches='tight', dpi=300)
