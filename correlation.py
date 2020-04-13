import cufflinks as cf
import plotly.graph_objects as go
import plotly.subplots as sb


import numpy as np
import pandas as pd

import itertools as it



def sort_by_corr(df_corr: pd.DataFrame) -> pd.DataFrame:
    """Sort the correlation df of stock prices by the median
       correlation value of each stock.
    """
    df_corr_sort = df_corr.copy(deep=True)
    stock_corr_ixes = df_corr_sort.median().argsort()
    # sort rows
    df_corr_sort = df_corr_sort.iloc[stock_corr_ixes]
    # sort columns
    df_corr_sort = df_corr_sort.transpose()
    return df_corr_sort.iloc[stock_corr_ixes]



def find_top_corr_pairs(
    df_corr_sort: pd.DataFrame,
    num_top_corrs: int,
    outlier_percentile: float,
) -> pd.DataFrame:
    """From a sorted correlation df of stock prices, find the top most correlated stocks.
    
    :param outlier_percentile: remove the top and bottom of this percentile correlations.
    
    Sample input::
    
        find_top_corr_pairs(df_corr_sort=df_corr_sort, num_top_corrs=2, outlier_percentile=1)

    Sample output::
    
          stock_1 stock_2      corr
        0     AMD    PRGO -0.728557
        1       C      LB -0.728555
        2    ALLE       V  0.950137
        3    ADSK     JPM  0.950147
    """
    # Flatten and sort corr values, remove nans and corr=1 (same stock).
    corrs = df_corr_sort.values.flatten()
    corrs = corrs[(~np.isnan(corrs)) & (corrs<1)]
    corrs = pd.Series(np.sort(corrs))
    min_corr, max_corr = corrs.quantile(q=outlier_percentile/100), corrs.quantile(q=(100-outlier_percentile)/100)
    corrs = corrs[(corrs >= min_corr) & (corrs <= max_corr)]
    
    # Find top pos and neg corr values
    top_neg_corrs = set(corrs[:(num_top_corrs*2)])
    top_pos_corrs = set(corrs[-(num_top_corrs*2):])
    top_corr_vals = top_pos_corrs | top_neg_corrs
    
    # Fetch the row and col with the top corr values
    df_top_pairs = df_corr_sort.copy(deep=True)
    # v = array([['AAL', 'AAL', 1.0],
    #        ['ALK', 'AAL', 0.6700483846046383],
    #        ['AWK', 'AAL', 0.663461997140519],
    #        ...,
    #        ['VRTX', 'XEL', 0.9451115380381667],
    #        ['WEC', 'XEL', 0.9912863304334782],
    #        ['XEL', 'XEL', 1.0]], dtype=object)
    v = df_top_pairs.unstack().to_frame().sort_index(level=1).reset_index().values
    top_pairs = {tuple(sorted([s1, s2])): corr for (s1, s2, corr) in v if s1!=s2 and corr in top_corr_vals}
    return pd.DataFrame(
        [(s1, s2, corr) for (s1, s2), corr in top_pairs.items()],
        columns=['stock_1', 'stock_2', 'corr']
    ).sort_values(by='corr').reset_index(drop=True)


def plot_top_corr_stocks(
    price_df: pd.DataFrame,
    top_corr_stocks_df: pd.DataFrame,
    num_to_plot: int,
    min_dt: pd.Timestamp,
    height = None,
    width = None,
) -> None:
    """Plots mean centered prices timeseries for top correlated stocks.
    
    :param top_corr_stocks_df: Output of :func:`find_top_corr_pairs`.
    """
    rows = list(
        it.chain(
            # top neg corr
            top_corr_stocks_df.iloc[:num_to_plot].iterrows(),
            # top pos corr
            top_corr_stocks_df.iloc[-num_to_plot:].iterrows(),
        ),
    )
    fig = sb.make_subplots(
        rows=num_to_plot,
        cols=2,
        subplot_titles=[f'{r.stock_1} vs {r.stock_2} mean centered' for _, r in rows]
    )
    
    for row_num, col_num, (_, r) in zip(
        list(range(1, num_to_plot+1)) * 2,
        ([1] * num_to_plot) + ([2] * num_to_plot),
        rows,
    ):
        # Mean center
        s1, s2 = r.stock_1, r.stock_2
        df_small = price_df[min_dt:][[s1, s2]]
        df_small[s1] = df_small[s1] - df_small[s1].mean()
        df_small[s2] = df_small[s2] - df_small[s2].mean()
        
        fig.add_trace(
            go.Scatter(x=df_small.index, y=df_small[s1], name=s1),
            row=row_num,
            col=col_num,
        )
        fig.add_trace(
            go.Scatter(x=df_small.index, y=df_small[s2], name=s2),
            row=row_num,
            col=col_num,
        )
    fig.update_layout(
        height=height or 800,
        width=width or 1200,
        title_text="Comparision of top correlated stocks: Left negatively, Right positively correlated",
    )
    fig.show()