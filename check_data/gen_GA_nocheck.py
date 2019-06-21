'''
将evaluate_new 文件进行重新生成只有zid layout 不用碰撞
'''
import pandas as pd
import csv
import json

path = r'F:\pycharm\WorkSpace\GA\GA_functionZone\data\evaluate_new.csv'
gen_path = r'F:\pycharm\WorkSpace\GA\GA_functionZone\data\GA_nocheck.csv'
df = pd.read_csv(path)
zids = [json.loads(x) for x in df['zids']]
layouts = [json.loads(x) for x in df['layout_pairs']]
data =['zids','layouts']
with open(gen_path, 'a', newline='') as t_file:
    csv_writer = csv.writer(t_file)
    csv_writer.writerow(data)
    t_file.close()
for i in range(df.shape[0]):
    print('正在读取第{}行'.format(i))
    zid = df.zids.iloc[i]
    for k in range(2):
        layout = layouts[i][k]
        data =[zid,layout]
        with open(gen_path, 'a', newline='') as t_file:
            csv_writer = csv.writer(t_file)
            csv_writer.writerow(data)
            t_file.close()
