# import matplotlib.pyplot as plt
# import numpy as np
#
# def huatu(aac1):
#     n_groups = 2
#
#     # means_men = (0.9829, 0.9881, 0.978, 0.9657)
#     # std_men = (1, 1, 1, 1, 1)
#
#     # means_women = (0.89, 0.87, 0.92, 0.79)
#     # std_women = (3, 5, 2, 3, 3)
#     x_label = ['positive and predicted ATP-BPs', 'negative and predicted ATP-BPs']
#     fig, ax = plt.subplots()
#
#     index = np.arange(n_groups)
#     # index = [0, 0.2]
#     print(index)
#     bar_width = 0.15
#
#     opacity = 0.4
#     error_config = {'ecolor': '0.3'}
#
#     rects1 = ax.bar(index, aac1, bar_width,
#                     alpha=opacity, color='b',
#                     error_kw=error_config,
#                     label="Mbah's method" )
#
#
#
#     # plt.ylim([0, 0.15])
#     plt.ylim([0.2, 1.2])
#     ax.set_title("Comparison with Hou's method")
#     ax.set_xticks(index + bar_width, x_label)
#     # ax.set_xticklabels(('positive and predicted ATP-BPs', 'negative and predicted ATP-BPs'))
#     ax.legend()
#     fig.tight_layout()
#     plt.show()
#
# def main():
#     acc1 = [0.8113, 0.8469]
#
#     huatu(acc1)
#
# if __name__ == '__main__':
#     main()

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 防止乱码
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# mpl.rcParams["axes.unicode_minus"] = False
#
# # 生成数据
# x = np.arange(2)
# y = [0.061,0.079]
# # y1 = [2,6,3,8,5]
#
# bar_width = 0.1
# tick_label = ["positive and predicted","negative and predicted"]
#
# # 生成多数据并列柱状图
# plt.bar(x + bar_width/2,y,bar_width,color="c",align="center",alpha=0.5)
# # plt.bar(x+bar_width,y1,bar_width,color="b",align="center",label="班级B",alpha=0.5)
# # 生成多数据平行柱状图
# # plt.barh(x,y,bar_width,color="c",align="center",label="班级A",alpha=0.5)
# # plt.barh(x+bar_width,y1,bar_width,color="b",align="center",label="班级B",alpha=0.5)
# plt.ylim([0.05, 0.08])
#
# # 设置x,y轴标签
# # plt.xlabel("测试难度")
# # plt.ylabel("试卷份数")
#
# # 设置x轴标签位置
# plt.xticks(x + bar_width/2,tick_label)
#
# plt.legend()
#
# plt.show()
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 防止乱码
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

# 生成数据
x = np.arange(2)
y = [0.061, 0.079]


bar_width = 0.2
tick_label = ["positive and\n predicted","negative and\n  predicted"]

# 生成多数据并列柱状图
# plt.bar(x,y,bar_width,color="c",align="center",label="班级A",alpha=0.5)
# plt.bar(x+bar_width,y1,bar_width,color="b",align="center",label="班级B",alpha=0.5)
# 生成多数据平行柱状图
plt.barh(x+bar_width,y,bar_width,color="b",align="center",alpha=0.4)
# plt.barh(x+bar_width,y1,bar_width,color="b",align="center",label="班级B",alpha=0.5)

plt.xlim([0.05, 0.08])
# 设置x轴标签位置
plt.yticks(x+bar_width,tick_label)
# ax = plt.subplots()
# ax.set_title()
plt.legend()
plt.title("The gap of  average evolutionary distances")

plt.show()