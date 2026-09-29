# Survivorship bias check

*Biased*: the 300 constituents at the end of the sample, treated as the universe for the whole period. *Point-in-time*: every stock only while it was actually a CSI 300 member (default pipeline). Same factors, preprocessing, costs and in-sample sign fitting. L/S = Q5 − Q1, gross, annualized.

| Factor | Dir biased | Dir PIT | RankIC biased | RankIC PIT | t biased | t PIT | L/S ann. biased | L/S ann. PIT | L/S Sharpe biased | L/S Sharpe PIT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mom_20 | − | − | -0.015 | -0.021 | -1.302 | -2.102 | -0.0% | 2.5% | 0.098 | 0.227 |
| mom_120_skip20 | + | + | 0.008 | 0.008 | 0.637 | 0.699 | 3.1% | -0.2% | 0.247 | 0.089 |
| rev_5 | + | + | 0.007 | 0.009 | 1.113 | 1.479 | 0.8% | 0.1% | 0.135 | 0.087 |
| vol_20 | + | − | -0.020 | -0.038 | -1.644 | -3.251 | 10.2% | 1.0% | 0.538 | 0.146 |
| turn_20 | − | − | -0.030 | -0.037 | -3.047 | -3.742 | -4.4% | 8.3% | -0.098 | 0.529 |
| abn_turn | − | − | -0.008 | -0.012 | -0.809 | -1.471 | -3.4% | -3.3% | -0.188 | -0.219 |
| amihud_20 | + | + | 0.024 | 0.020 | 3.105 | 2.618 | 7.7% | 6.7% | 0.576 | 0.565 |
| pv_corr_20 | − | − | -0.030 | -0.027 | -3.177 | -3.095 | 10.7% | 11.2% | 0.755 | 0.824 |
| range_20 | + | − | -0.025 | -0.045 | -2.079 | -3.822 | 5.8% | 3.9% | 0.353 | 0.286 |
| size | − | + | -0.006 | 0.030 | -0.928 | 3.167 | 22.5% | 4.3% | 1.283 | 0.335 |
| composite | + | + | 0.030 | 0.035 | 3.177 | 4.113 | 10.7% | 7.7% | 0.755 | 0.565 |
