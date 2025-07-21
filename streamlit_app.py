# streamlit_app.py
import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
# 初始化ETF数据库（A股+港股）
ETF_DATABASE = {
    "上证50ETF": "510050",
    "沪深300ETF": "510300",
    "中证500ETF": "510500",
    "中证1000ETF": "512100",
    "中证2000ETF": "563300",
    "创50ETF": "159681",
    "银行ETF": "512800",
    "黄金ETF": "518880",
    "港股创新药ETF": "513120",
    "港股互联网ETF": "513770",
    "科创芯片50ETF": "588200",
    "机器人产业ETF": "560630",
    "通信ETF": "515880",
    "工业有色ETF": "560860",
    "交运ETF": "561320",
    "公用事业ETF": "560190",
    "港股通非银ETF": "513750",
    "金融科技ETF": "516860",
    "半导体ETF": "512480",
    "新能源ETF": "516160",
    "人工智能AIETF": "515070",
    "消费ETF": "510150",
    "酒ETF": "512690",
    "军工龙头ETF": "512710",
    "证券ETF": "512880",
    "医药ETF": "512010",
    "恒生医疗ETF": "513060",
    "消费电子ETF": "561600",
    "电力ETF": "561560",
    "房地产ETF": "512200",
    "基建ETF": "516950"
}
# 动量得分计算模型
@st.cache_data(ttl=600)  # 缓存10分钟
def calculate_momentum_scores(df, date, trend_window=25):
    """
    计算ETF三大核心因子得分
    :param df: 包含OHLCV数据的DataFrame
    :param date: 指定评估日期
    :return: 字典格式的评分结果
    """
    # 筛选指定日期前的数据
    df_sub = df[df.index <= date].iloc[-trend_window * 2:]
    if len(df_sub) < trend_window:
        return {"错误": "数据不足"}
    # 1. 趋势强度因子（线性回归斜率+R²）
    x = np.arange(len(df_sub))
    y = np.log(df_sub['close'])
    slope, _, r_value, _, _ = stats.linregress(x, y)
    trend_score = (slope * 250) * (r_value ** 2)  # 年化斜率×R平方
    # 2. 动量因子（5日+10日收益率）
    roc_5 = (df_sub['close'].iloc[-1] / df_sub['close'].iloc[-6] - 1) * 100
    roc_10 = (df_sub['close'].iloc[-1] / df_sub['close'].iloc[-11] - 1) * 100
    momentum_score = 0.6 * roc_5 + 0.4 * roc_10  # 短期动量加权
    # 3. 量能因子（成交量均线比）
    vol_ma_short = df_sub['volume'].rolling(5).mean().iloc[-1]
    vol_ma_long = df_sub['volume'].rolling(20).mean().iloc[-1]
    volume_score = np.log(vol_ma_short / vol_ma_long) if vol_ma_long > 0 else 0
    # 综合得分（归一化到0-100分）
    total_score = 40 * trend_score + 35 * momentum_score + 25 * volume_score
    return {
        '趋势强度': round(trend_score, 2),
        '动量得分': round(momentum_score, 2),
        '量能指标': round(volume_score, 2),
        '综合评分': max(0, min(100, round(total_score, 2)))
    }
# 获取历史数据（AKShare接口）
@st.cache_data(ttl=600)
def fetch_etf_data_ak(symbol, start_date):
    """适配A股/港股ETF数据获取规则"""
    try:
        # 使用AKShare获取ETF历史数据
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="qfq")
        # 列名标准化处理
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        })
        # 日期处理
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        # 筛选日期范围
        return df[df.index >= pd.to_datetime(start_date)]
    except Exception as e:
        st.error(f"数据获取失败: {str(e)}")
        return pd.DataFrame()
# 使用Plotly生成K线图
def generate_plotly_chart(df, days=60):
    """生成带移动平均线的K线图（使用Plotly）"""
    df = df.tail(days).copy()
    # 确保数据格式正确
    if 'close' not in df.columns:
        st.error("数据格式错误，缺少'close'列")
        return None
    # 计算移动平均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    # 创建子图：主图为K线图，副图为成交量
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    # 添加K线图
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='#ef5350',  # 上涨红色
            decreasing_line_color='#26a69a'  # 下跌绿色
        ),
        row=1, col=1
    )
    # 添加5日均线
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MA5'],
            name='5日均线',
            line=dict(color='#1f77b4', width=1.5),
            opacity=0.8
        ),
        row=1, col=1
    )
    # 添加20日均线
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MA20'],
            name='20日均线',
            line=dict(color='#ff7f0e', width=1.5),
            opacity=0.8
        ),
        row=1, col=1
    )
    # 添加成交量柱状图
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='成交量',
            marker_color='#7f7f7f',
            opacity=0.6
        ),
        row=2, col=1
    )
    # 设置布局
    fig.update_layout(
        title=f'最近{days}个交易日走势',
        xaxis_title='日期',
        yaxis_title='价格',
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        height=600,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    # 设置Y轴标题
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    # 禁用范围选择器（rangeselector）
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig
# 主应用界面
def app():
    # 标题和说明
    st.title("📊 ETF动量评分与可视化系统")
    # 日期选择器
    max_date = datetime.now() - timedelta(days=1)
    selected_date = st.date_input(
        "选择评估日期",
        value=max_date,
        max_value=max_date
    )
    # ETF多选
    selected_etfs = st.multiselect(
        "选择ETF",
        options=list(ETF_DATABASE.keys()),
        default=["银行ETF", "港股创新药ETF","沪深300ETF"]
    )
    # 高级参数
    with st.expander("高级设置"):
        trend_window = st.slider(
            "趋势计算窗口(日)",
            min_value=20,
            max_value=60,
            value=25
        )
        # 数据范围
        start_date = st.date_input(
            "数据开始日期",
            value=selected_date - timedelta(days=365)
        )
        # 权重调整
        st.markdown("**因子权重调整**")
        trend_weight = st.slider("趋势强度权重", 0, 100, 40)
        momentum_weight = st.slider("动量得分权重", 0, 100, 35)
        volume_weight = st.slider("量能指标权重", 0, 100, 25)
        # 缓存控制
        st.caption(f"当前缓存状态: {len(st.session_state)}")
        if st.button("清除缓存"):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()
    # 主内容区
    if st.button("生成分析报告", type="primary", use_container_width=True):
        if not selected_etfs:
            st.warning("请至少选择一个ETF进行分析")
            return
        results = []
        charts = []
        # 遍历选中的ETF
        progress_bar = st.progress(0)
        for i, etf_name in enumerate(selected_etfs):
            progress = (i + 1) / len(selected_etfs)
            progress_bar.progress(progress, text=f"处理 {etf_name}...")
            # 获取数据
            symbol = ETF_DATABASE[etf_name]
            df = fetch_etf_data_ak(symbol, start_date.strftime("%Y-%m-%d"))
            if df.empty:
                st.warning(f"{etf_name}({symbol}) 数据获取失败，跳过")
                continue
            # 计算动量得分
            scores = calculate_momentum_scores(df, selected_date.strftime("%Y-%m-%d"), trend_window)
            # 动态调整权重
            total_score = (
                    trend_weight * scores["趋势强度"] +
                    momentum_weight * scores["动量得分"] +
                    volume_weight * scores["量能指标"]
            )
            scores["综合评分"] = max(0, min(100, round(total_score, 2)))
            # 生成Plotly图表
            fig = generate_plotly_chart(df)
            # 存储结果
            results.append({
                "ETF": etf_name,
                "代码": symbol,
                **scores
            })
            if fig:
                charts.append(fig)
        progress_bar.empty()
        if not results:
            st.error("所有ETF数据获取失败，请检查网络连接或代码配置")
            return
        # 展示评分结果表格
        st.subheader("📝 ETF动量评分结果")
        df_results = pd.DataFrame(results)
        df_results["推荐权重"] = df_results["综合评分"] / df_results["综合评分"].sum()
        # 高亮显示最佳ETF
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #a1d99b' if v else '' for v in is_max]
        st.dataframe(
            df_results.style
            .apply(highlight_max, subset=["综合评分"])
            .format({"推荐权重": "{:.2%}"}),
            height=min(600, 45 * len(df_results))
        )
        # 展示最佳ETF建议
        best_etf = df_results.loc[df_results["综合评分"].idxmax(), "ETF"]
        best_weight = df_results.loc[df_results["综合评分"].idxmax(), "推荐权重"]
        st.success(f"**策略建议**：优先配置 **{best_etf}**，建议仓位 **{best_weight:.1%}**")
        # 展示Plotly图表
        if charts:
            st.subheader("📈 K线趋势分析 (交互式图表)")
            for idx, fig in enumerate(charts):
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"{selected_etfs[idx]} 技术图表（最近60个交易日）")
        # 数据导出选项
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="下载评分数据(CSV)",
                data=df_results.to_csv(index=False).encode("utf-8"),
                file_name=f"etf_scores_{selected_date}.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("查看实时行情", use_container_width=True):
                try:
                    spot_data = ak.fund_etf_spot_em()
                    st.dataframe(
                        spot_data[["代码", "名称", "最新价", "涨跌幅", "成交量"]]
                        .sort_values("涨跌幅", ascending=False)
                        .head(10)
                    )
                except Exception as e:
                    st.error(f"实时行情获取失败: {str(e)}")
# 应用入口
if __name__ == "__main__":
    # # 页面配置
    # st.set_page_config(
    #     page_title="ETF动量评分系统",
    #     layout="wide",
    #     page_icon="📈"
    # )
    app()
