论文图表与原始数据（由 paper/export_figs.py 自动生成，未经手工修改）
======================================================================

runs_all.csv          全部真实演练逐局明细（14 列，字段名即含义）
runs_summary.csv      按批次汇总：局数/平均清除/平均测到频道/缺口/空挥率/平均虚拟时间
table_tests.csv       三个案例的成绩表（清除数、平均定位清除时间、程序运行时间）
coverage_stations.csv 覆盖站点坐标与覆盖半径（由 dog/routing.py::greedy_cover 生成）
traj_<run_id>.csv     轨迹点序列：seq/虚拟时间/x/y/动作/频道/结果
events_<run_id>.csv   事件时间线：seq/虚拟时间/频道/动作/结果/坐标/示向度
fig_coverage.svg      覆盖站点与覆盖圆示意图
fig_flowchart.svg     算法流程图
traj_<run_id>.svg     机器狗轨迹图（含测得示向度、清除成功/失败标记）
events_<run_id>.svg   各频道事件时间线（纵轴频道，横轴虚拟时间）

说明：图为纯标准库生成的 SVG 矢量图，可直接放入 Word/LaTeX 或重新配色；
      所有图都由同目录 CSV 生成，保证图与正文数值同源。
