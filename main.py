# MAESC check
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

x = np.array([i for i in range(2, 60)])/2  # input for (60-2)/2 = 29 days twice a day
# Mood, Activity, Energy, Symptoms amd Mind clearness respectively (MAESC parameters)
y1 = np.array(np.random.randint(1, high=11, size=58))  # random input due to the scale from 1 to 10
y2 = np.array(np.random.randint(1, high=11, size=58))
y3 = np.array(np.random.randint(1, high=11, size=58))
y4 = np.array(np.random.randint(1, high=11, size=58))
y5 = np.array(np.random.randint(1, high=11, size=58))

plt.figure(figsize=(52, 43), dpi=100)
plt.subplots_adjust(hspace=0.5, left=0.03, right=0.97, bottom=0.04, top=0.96)


# Condition function 1. version
def condition(m, s, a):
    if a >= 5.25 and ((8 <= m <= 10 and s <= 4.5) or (7 <= m < 8 and s <= 3.5) or (5.5 <= m < 7 and s <= 3)):
        return 3
    elif a >= 4.5 and ((8 <= m <= 10 and s <= 5.75) or (7 <= m < 8 and s <= 5.5) or (6 <= m < 7 and s <= 4.5) or (5 <= m < 6 and s <= 4) or (4 <= m < 5 and s <= 3.25)):
        return 2
    elif a >= 4 and ((8 <= m <= 10 and s <= 7) or (7 <= m < 8 and s <= 6.5) or (6 <= m < 7 and s <= 6) or (5 <= m < 6 and s <= 5.5) or (4 <= m < 5 and s <= 4.25)):
        return 1
    else:
        return 0


# Condition function 2. version
def condition_uc(m, s, a):
    uc = 0
    conditions = [[[8, 10.1, 4.5, 5.25], [7, 8, 3.5, 5.25], [5.5, 7, 3, 5.25]], [[8, 10.1, 5.75, 4.5], [7, 8, 5.5, 4.5], [6, 7, 4.5, 4.5], [5, 6, 4, 4.5], [4, 5, 3.25, 4.5]], [[8, 10.1, 7, 4], [7, 8, 6.5, 4], [6, 7, 6, 4], [5, 6, 5.5, 4], [4, 5, 4.25, 4]]]
    if condition(m, s, a) == 3:
        for c in conditions[0]:
            if c[0] <= m < c[1]:
                uc += 1.25 * (c[2] - s) + 0.8*(a - c[3])
                break
        ucr = uc/5.25
        if ucr >= 0:
            return round(3+ucr, 3)
        else:
            return round(1+ucr+2, 3)
    elif 0 < condition(m, s, a) < 3:
        for c in conditions[3-condition(m, s, a)]:
            if c[0] <= m < c[1]:
                uc += 1.25*(c[2] - s) + 0.8*(a - c[3])
                break
        ucr = uc/5.25
        if ucr >= 0:
            return round(condition(m, s, a)+ucr, 3)
        else:
            return round(condition(m, s, a)+ucr, 3)
    else:
        if conditions[2][-1][0] <= m < conditions[2][0][1]:
            for c in conditions[2]:
                if c[0] <= m < c[1]:
                    uc += 1.25*(c[2] - s) + 0.8*(a - c[3])
                    break
        else:
            uc += 2*(m - conditions[2][-1][0]) + 1.25*(conditions[2][-1][2] - s) + 0.8*(a - conditions[2][-1][3])
        ucr = uc/5.25
        if ucr >= 0:
            return round(1-ucr, 3)
        elif -1 < ucr < 0:
            return round(ucr+1, 3)
        else:
            return -round(-ucr-1, 3)


# Activity-energy function
def acten(a, e):
    if a-0.5 <= e <= a+0.5:
        return 0
    elif e > a+1:
        return -1  # low activity/high energy
    else:
        return 1  # high activity/low energy


x_int = np.array(x, dtype=int)
day = len(np.unique(x_int))  # day, not the entries` amount

# date list generator
dates_pr = []
calendar = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
d = 20  # the day of the 1. entry
m = 10  # the month of the 1. entry
l = np.setdiff1d(np.where(x_int==x, x_int, 0), 0)
del_dates = []
for j in range(len(l)-1):
    if (l[j+1]-l[j]) > 1:
        del_dates.append([j, int(l[j+1]-l[j])])
for i in range(int(max(x))):
    if m < 12:
        if d <= calendar.get(m):
            dates_pr.append(str(d)+'/'+str(m))
            d += 1
        else:
            d = 1
            m += 1
            dates_pr.append(str(d) + '/' + str(m))
            d += 1
    elif m == 12:
        if d <= calendar.get(m):
            dates_pr.append(str(d)+'/'+str(m))
            d += 1
        else:
            d = 1
            m = 1
            dates_pr.append(str(d) + '/' + str(m))
            d += 1
dates_pr.append(str(d) + '/' + str(m))

dates_np = dates_pr.copy()
for j in range(len(del_dates)):
    dates_np = np.delete(dates_np, [[i for i in range(del_dates[j][0]+1, del_dates[j][0]+del_dates[j][1])]])

dates = []
for i in range(len(dates_np)):
    dates.append(str(dates_np[i]))  # dates adjusted for skipped days

yy1 = np.where(x_int==x, y1, 11)
yy2 = np.where(x_int==x, y2, 11)
yy3 = np.where(x_int==x, y3, 11)
yy4 = np.where(x_int==x, y4, 11)
yy5 = np.where(x_int==x, y5, 11)
con = np.array([])
for i in range(len(x)):
    con = np.append(con, condition(y1[i], y4[i], y2[i]))  # con = condition for entries

weigts_con = [0.5 for j in range(len(x))]
for i in range(len(x)-1):
    if x[i+1] != x[i]+0.5 and (x[i] % 2 == 0.0 or x[i] % 2 == 1.0):
        weigts_con[i] = 1
days_cond = np.unique(x_int)

#  mean values calculation
m1 = np.average(y1, weights=weigts_con)
m2 = np.average(y2, weights=weigts_con)
m3 = np.average(y3, weights=weigts_con)
m4 = np.average(y4, weights=weigts_con)
m5 = np.average(y5, weights=weigts_con)
# standard deviation calculation for Mood and Symptoms
s1 = np.std(y1)
s4 = np.std(y4)

conddd = np.where(x_int==x, con, 5)  # mean condition daily with zeros (5)
morn = np.setdiff1d(np.where(x_int==x, con, 5), 5, assume_unique=True)  # condition values in the morning
eve = np.setdiff1d(np.where(x_int!=x, con, 5), 5, assume_unique=True)  # condition values in the evening

# mean condition and mean MAESC parameters daily calculation
for i in range(len(x)):
    if conddd[i] == 5:
        conddd[i - 1] = condition((y1[i]+y1[i-1])/2, (y4[i]+y4[i-1])/2, (y2[i]+y2[i-1])/2)  # conddd = condition for days with zeros (5)
        yy1[i - 1] = (y1[i] + y1[i - 1]) / 2
        yy2[i - 1] = (y2[i] + y2[i - 1]) / 2
        yy3[i - 1] = (y3[i] + y3[i - 1]) / 2
        yy4[i - 1] = (y4[i] + y4[i - 1]) / 2
        yy5[i - 1] = (y5[i] + y5[i - 1]) / 2
condd = np.setdiff1d(conddd, 5, assume_unique=True)  # condd = condition daily without zeros (5)
# MAESC daily values
y1_d = np.setdiff1d(yy1, 11, assume_unique=True)
y2_d = np.setdiff1d(yy2, 11, assume_unique=True)
y3_d = np.setdiff1d(yy3, 11, assume_unique=True)
y4_d = np.setdiff1d(yy4, 11, assume_unique=True)
y5_d = np.setdiff1d(yy5, 11, assume_unique=True)

morning = np.copy(morn)  # calculation of true morning condition values
del_morn = []
i = 0
j = 0
k = 0
del_change = []
while i < len(x)-1:
    if x[i+1] != x[i]+0.5 and (x[i] % 2 == 0.0 or x[i] % 2 == 1.0):
        morning = np.delete(morning, j)
        del_morn.append(i)
        del_change.append(k)
        k += 1
    elif x[i] % 2 == 0.5 or x[i] % 2 == 1.5:
        j += 1
    else:
        k += 1
    i += 1
if x[-1] % 2 == 0.0 or x[-1] % 2 == 1.0:
    morning = np.delete(morning, -1)
    del_morn.append(len(morning)-1)
del_morning = np.append(del_morn, len(dates)-1)

# condition changes calculation (morning condition to evening condition)
change_0 = np.where(morning == eve, 0, 5)
change_1_n = np.where(morning-1 == eve, -1, 5)
change_2_n = np.where(morning-2 == eve, -2, 5)
change_3_n = np.where(morning-3 == eve, -3, 5)
change_1 = np.where(morning+1 == eve, 1, 5)
change_2 = np.where(morning+2 == eve, 2, 5)
change_3 = np.where(morning+3 == eve, 3, 5)
changes = [change_0, change_1_n, change_2_n, change_3_n, change_1, change_2, change_3]
change = []
for j in range(len(change_0)):
    for i in changes:
        if i[j]!=5:
            change.append(int(i[j]))

change = np.array(change)
count_pos = len(np.setdiff1d(change_1, 5, True))+len(np.setdiff1d(change_2, 5, True))+len(np.setdiff1d(change_3, 5, True))
count_neg = len(np.setdiff1d(change_1_n, 5, True))+len(np.setdiff1d(change_2_n, 5, True))+len(np.setdiff1d(change_3_n, 5, True))
count_pos_3 = len(np.setdiff1d(change_3, 5, True))
count_neg_3 = len(np.setdiff1d(change_3_n, 5, True))
count_pos_2 = len(np.setdiff1d(change_2_n, 5, True))
count_neg_2 = len(np.setdiff1d(change_2_n, 5, True))
count_pos_1 = len(np.setdiff1d(change_1, 5, True))
count_neg_1 = len(np.setdiff1d(change_1_n, 5, True))
count_zero = len(np.setdiff1d(change_0, 5, True))

c_m = np.average(con, weights=weigts_con)  # weighted condition average/mean value (for entries)
c_s = np.std(con)
cd_m = np.mean(condd)
cd_s = np.std(condd)
ch_m = np.mean(change)  # mean value for condition changes
ch_s = np.std(change)

x_change = np.delete(days_cond, del_change)  # days when changes are possible to calculate
if x[-1]%2==0.0 or x[-1]%2==1.0:
    x_change = np.delete(x_change, -1)

cnd = np.array([condition_uc(y1[i], y4[i], y2[i]) for i in range(len(x))])  # condition according to 2. version function

# bad, danger, ok and good days according to condition
bad = np.where(condd == 0, 1, 0)
danger = np.where(condd == 1, 1, 0)
ok = np.where(condd == 2, 1, 0)
good = np.where(condd == 3, 1, 0)

bad_c = np.where(con == 0, 1, 0)
danger_c = np.where(con == 1, 1, 0)
ok_c = np.where(con == 2, 1, 0)
good_c = np.where(con == 3, 1, 0)

# Activity-energy function into list of values
counter_acten = []
acten_data = []
for i in range(0, day):
    acten_data.append(acten(y2_d[i], y3_d[i]))

# weekly values
mean_MAESC = []
std_MAESC = []
mean_con = []
std_con = []
mean_cnd = []
std_cnd = []
mean_change = []
std_change = []
cu_m = float(np.average(cnd, weights=weigts_con))

for i in range((x_int[-1]-1)//7):
    mean_MAESC.append([float(np.mean(y1[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])), float(np.mean(y2[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])), float(np.mean(y3[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])), float(np.mean(y4[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])), float(np.mean(y5[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1]))])
    mean_con.append(float(np.average(con[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1], weights=weigts_con[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])))
    mean_change.append(float(np.mean(change[np.where(x_change>=(i*7+1))[0][0]:np.where(x_change<=(i+1)*7)[0][-1]+1])))
    std_MAESC.append([float(np.std(y1[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])), float(np.std(y2[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])), float(np.std(y3[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])), float(np.std(y4[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])), float(np.std(y5[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1]))])
    std_con.append(float(np.std(con[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])))
    std_change.append(float(np.std(change[np.where(x_change>=(i*7+1))[0][0]:np.where(x_change<=(i+1)*7)[0][-1]+1])))
    counter_acten.append(np.count_nonzero(acten_data[np.where(np.unique(x_int)>=(i*7+1))[0][0]:np.where(np.unique(x_int)<=(i+1)*7)[0][-1]+1]))
    mean_cnd.append(float(np.average(cnd[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1], weights=weigts_con[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])))
    std_cnd.append(float(np.std(cnd[np.where(x_int>=(i*7+1))[0][0]:np.where(x_int<=(i+1)*7)[0][-1]+1])))

# Alert calculation -> week evaluation
c = [0 for j in range((x_int[-1]-1)//7)]
cc = [0 for j in range((x_int[-1]-1)//7)]
for week in range((x_int[-1]-1)//7):
    if mean_MAESC[week][0] < 5.35:
        c[week] += 1
    elif 5.35 <= mean_MAESC[week][0] < 6:
        cc[week] += 1
    if mean_MAESC[week][1] < 5.25:
        c[week] += 1
    elif 5.25 <= mean_MAESC[week][1] < 5.8:
        cc[week] += 1
    if mean_MAESC[week][2] < 5:
        c[week] += 1
    elif 5 <= mean_MAESC[week][2] < 5.4:
        cc[week] += 1
    if mean_MAESC[week][3] >= 5.8:
        c[week] += 1
    elif 5.8 > mean_MAESC[week][3] >= 5.1:
        cc[week] += 1
    if mean_MAESC[week][4] < 5.8:
        c[week] += 1
    elif 5.8 <= mean_MAESC[week][4] < 6.25:
        cc[week] += 1
    if std_MAESC[week][0] > 1.5:
        c[week] += 1
    elif 1.5 >= std_MAESC[week][0] > 1.2:
        cc[week] += 1
    if std_MAESC[week][3] >= 2:
        c[week] += 1
    elif 2 > std_MAESC[week][3] > 1.69:
        cc[week] += 1
    if mean_con[week] < 1.5:
        c[week] += 0.5
    elif 1.5 <= mean_con[week] < 1.7:
        cc[week] += 0.5
    if std_con[week] > 0.95:
        c[week] += 0.5
    elif 0.95 >= std_con[week] > 0.8:
        cc[week] += 0.5
    if mean_cnd[week] < 1.5:
        c[week] += 0.5
    elif 1.5 <= mean_cnd[week] < 1.7:
        cc[week] += 0.5
    if std_cnd[week] > 0.95:
        c[week] += 0.5
    elif 0.95 >= std_cnd[week] > 0.8:
        cc[week] += 0.5
    if mean_change[week] > 0.5 or mean_change[-1] < -0.5:
        c[week] += 1
    elif 0.5 >= mean_change[week] > 0.4 or -0.5 <= mean_change[-1] < -0.4:
        cc[week] += 1
    if std_change[week] > 1:
        c[week] += 1
    elif 1 >= std_change[week] > 0.88:
        cc[week] += 1
    if counter_acten[week] > 3:
        c[week] += 1
    elif 3 >= counter_acten[week] > 2:
        cc[week] += 1

c_c = np.array([13-(c[j]+cc[j]) for j in range((x_int[-1]-1)//7)])  # green zone weekly
c = np.array(c)  # red zone weekly
cc= np.array(cc)  # yellow zone weekly


# Plots

plt.subplot(5, 3, 1)
plt.ylim(0, 10)
plt.xlim(0, max(x_int)+1)
plt.xticks(np.arange(1, max(x_int)+2, round(x_int[-1]/15)), dates_pr[::round(x_int[-1]/15)], fontsize=15)
plt.yticks(np.arange(0, 11, 1.0), fontsize=15)
plt.plot(x, y1, c='#3CB371', lw=4)
plt.axhline(6, c='r', ls='dashed', label='critical value')
plt.axhline(m1, c='g', ls='dashed', label='mean value')
plt.title('Mood', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 2)
plt.ylim(0, 10)
plt.xlim(0, max(x_int)+1)
plt.xticks(np.arange(1, max(x_int)+2, round(x_int[-1]/15)), dates_pr[::round(x_int[-1]/15)], fontsize=15)
plt.yticks(np.arange(0, 11, 1.0), fontsize=15)
plt.plot(x, y2, c='#FFA500', lw=4)
plt.axhline(5.8, c='r', ls='dashed', label='critical value')
plt.axhline(m2, c='g', ls='dashed', label='mean value')
plt.title('Activity', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 3)
plt.ylim(0, 10)
plt.xlim(0, max(x_int)+1)
plt.xticks(np.arange(1, max(x_int)+2, round(x_int[-1]/15)), dates_pr[::round(x_int[-1]/15)], fontsize=15)
plt.yticks(np.arange(0, 11, 1.0), fontsize=15)
plt.plot(x, y3, c='#FF1493', lw=4)
plt.axhline(5.4, c='r', ls='dashed', label='critical value')
plt.axhline(m3, c='g', ls='dashed', label='mean value')
plt.title('Energy', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 4)
plt.ylim(0, 10)
plt.xlim(0, max(x_int)+1)
plt.xticks(np.arange(1, max(x_int)+2, round(x_int[-1]/15)), dates_pr[::round(x_int[-1]/15)], fontsize=15)
plt.yticks(np.arange(0, 11, 1.0), fontsize=15)
plt.plot(x, y4, c='#7CFC00', lw=4)
plt.axhline(5.8, c='r', ls='dashed', label='critical value')
plt.axhline(m4, c='g', ls='dashed', label='mean value')
plt.title('Symptoms', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 5)
plt.ylim(0, 10)
plt.xlim(0, max(x_int)+1)
plt.xticks(np.arange(1, max(x_int)+2, round(x_int[-1]/15)), dates_pr[::round(x_int[-1]/15)], fontsize=15)
plt.yticks(np.arange(0, 11, 1.0), fontsize=15)
plt.plot(x, y5, c='#FFD700', lw=4)
plt.axhline(5.8, c='r', ls='dashed', label='critical value')
plt.axhline(m5, c='g', ls='dashed', label='mean value')
plt.title('Mind clearness', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 9)
plt.ylim(-3.5, 3.5)
plt.xlim(0, max(days_cond)+1)
plt.xticks(np.arange(1, max(days_cond)+2, round(len(days_cond)/15)), dates_pr[::round(len(days_cond)/15)], fontsize=15)
plt.yticks(fontsize=15)
plt.scatter(x_change, change, c = change, cmap = mcolors.ListedColormap(['#0000CD', '#1E90FF', '#00FFFF', 'g', '#FFA500', '#FF4500', '#B22222']), s=80, zorder=3)
plt.plot(x_change, change, c='#800000', lw=4)
plt.axhline(1.5, c='r', ls='dashed', label='critical values')
plt.axhline(-1.5, c='r', ls='dashed')
plt.axhline(ch_m, c='g', ls='dashed', label='mean value')
plt.title('Condition changes', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 7)
plt.ylim(-0.5, 3.5)
plt.yticks(np.arange(0.0, 4.0, 1.0), ['Bad', 'Danger', 'OK', 'Good'], fontsize=15)
plt.xlim(0, max(x_int)+1)
plt.xticks(np.arange(1, max(x_int)+2, round(x_int[-1]/15)), dates_pr[::round(x_int[-1]/15)], fontsize=15)
plt.plot(x, con, c='#00008B', lw=4)
plt.scatter(x, con, c = con, cmap = mcolors.ListedColormap(['r', '#FF8C00', '#FFD700', 'g']), s=70, zorder=2)
plt.axhline(1.5, c='r', ls='dashed', label='critical value')
plt.axhline(c_m, c='g', ls='dashed', label='mean value')
plt.title('Condition', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 8)
plt.ylim(-0.5, 3.5)
plt.yticks(np.arange(0.0, 4.0, 1.0), ['Bad', 'Danger', 'OK', 'Good'], fontsize=15)
plt.xlim(0, max(days_cond)+1)
plt.xticks(np.arange(1, max(days_cond)+2, round(len(days_cond)/15)), dates_pr[::round(len(days_cond)/15)], fontsize=15)
plt.plot(days_cond, condd, c='#00CED1', lw=4)
plt.scatter(days_cond, condd, c = condd, cmap = mcolors.ListedColormap(['r', '#FF8C00', '#FFD700', 'g']), s=80, zorder=2)
plt.axhline(1.5, c='r', ls='dashed', label='critical value')
plt.axhline(cd_m, c='g', ls='dashed', label='mean value')
plt.title('Condition', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 11)
labels = np.array(['Bad', 'Danger', 'OK', 'Good'])
yyyy = np.array([np.count_nonzero(bad)/day, np.count_nonzero(danger)/day, np.count_nonzero(ok)/day, np.count_nonzero(good)/day])
yyyyy = np.array([np.count_nonzero(bad_c)/len(x), np.count_nonzero(danger_c)/len(x), np.count_nonzero(ok_c)/len(x), np.count_nonzero(good_c)/len(x)])
colors = ['r', '#FF8C00', '#FFD700', 'g']
plt.pie(yyyy, radius=1, labels=labels, colors=colors, startangle = 180, autopct='%1.1f%%', pctdistance=0.8, textprops={'fontsize': 14}, wedgeprops=dict(width=0.4, edgecolor='w'))
plt.pie(yyyyy, radius=0.55, colors=colors, startangle = 180, autopct='%1.1f%%', textprops={'fontsize': 14}, wedgeprops=dict(width=0.4, edgecolor='w'))
plt.title('Condition percentage', fontsize=23)

plt.subplot(5, 3, 12)
llabels = np.array(['-3', '-2', '-1', '0', '1', '2', '3'])
y_c = np.array([count_neg_3, count_neg_2, count_neg_1, count_zero, count_pos_1, count_pos_2, count_pos_3])
colors = ['#0000CD', '#1E90FF', '#00FFFF', 'g', '#FFA500', '#FF4500', '#B22222']
plt.pie(y_c, labels=llabels, colors=colors, startangle = 180, autopct='%1.1f%%', textprops={'fontsize': 14}, wedgeprops=dict(width=0.85, edgecolor='w'))
plt.title('Condition changes', fontsize=23)

labelss = [dates_pr[::7][i]+'-'+dates_pr[6::7][i] for i in range((x_int[-1]-1)//7)]

plt.subplot(5, 3, 14)
plt.ylim(int(np.min(mean_MAESC)), int(np.max(mean_MAESC))+2)
plt.xlim(0.8, (x_int[-1]-1)//7+0.2)
plt.xticks(np.arange(1, (x_int[-1]-1)//7+1, 1), labelss, fontsize=15)
plt.yticks(fontsize=15)
plt.plot([i for i in range(1, (x_int[-1]-1)//7+1)], [mean_MAESC[j][0] for j in range((x_int[-1]-1)//7)], c='#3CB371', lw=2, label='Mood')
plt.plot([i for i in range(1, (x_int[-1]-1)//7+1)], [mean_MAESC[j][1] for j in range((x_int[-1]-1)//7)], c='#FFA500', lw=2, label='Activity')
plt.plot([i for i in range(1, (x_int[-1]-1)//7+1)], [mean_MAESC[j][2] for j in range((x_int[-1]-1)//7)], c='#FF1493', lw=2, label='Energy')
plt.plot([i for i in range(1, (x_int[-1]-1)//7+1)], [mean_MAESC[j][3] for j in range((x_int[-1]-1)//7)], c='#7CFC00', lw=2, label='Symptoms')
plt.plot([i for i in range(1, (x_int[-1]-1)//7+1)], [mean_MAESC[j][4] for j in range((x_int[-1]-1)//7)], c='#800080', lw=2, label='Mind Clearness')
plt.scatter([i for i in range(1, (x_int[-1]-1)//7+1)], [mean_MAESC[j][0] for j in range((x_int[-1]-1)//7)], c = '#3CB371', s=70, zorder=6)
plt.scatter([i for i in range(1, (x_int[-1]-1)//7+1)], [mean_MAESC[j][1] for j in range((x_int[-1]-1)//7)], c='#FFA500', s=70, zorder=6)
plt.scatter([i for i in range(1, (x_int[-1]-1)//7+1)], [mean_MAESC[j][2] for j in range((x_int[-1]-1)//7)], c='#FF1493', s=70, zorder=6)
plt.scatter([i for i in range(1, (x_int[-1]-1)//7+1)], [mean_MAESC[j][3] for j in range((x_int[-1]-1)//7)], c='#7CFC00', s=70, zorder=6)
plt.scatter([i for i in range(1, (x_int[-1]-1)//7+1)], [mean_MAESC[j][4] for j in range((x_int[-1]-1)//7)], c='#800080', s=70, zorder=6)
plt.title('Mean weakly MAESC', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 13)
plt.ylim(int(min(mean_change))-1, int(max(mean_con))+1)
plt.plot([i for i in range(1, (x_int[-1]-1)//7+1)], mean_con,  c='#00008B', lw=2, label='Condition')
plt.plot([i for i in range(1, (x_int[-1]-1)//7+1)], mean_change, c='#800000', lw=2, label='Condition changes')
plt.scatter([i for i in range(1, (x_int[-1]-1)//7+1)], mean_con, c = '#00008B', s=70, zorder=3)
plt.scatter([i for i in range(1, (x_int[-1]-1)//7+1)], mean_change, c='#800000', s=70, zorder=3)
plt.xticks(np.arange(1, (x_int[-1]-1)//7+1, 1), labelss, fontsize=15)
plt.yticks(fontsize=15)
plt.title('Mean weekly Condition', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 6)
plt.ylim(0, 13)
plt.bar([i for i in range((x_int[-1]-1)//7)], c, color='r', width=0.5-((x_int[-1]-1)//7-5)*0.025)
plt.bar([i for i in range((x_int[-1]-1)//7)], cc, bottom=c, color='#FF8C00', width=0.5-((x_int[-1]-1)//7-5)*0.025)
plt.bar([i for i in range((x_int[-1]-1)//7)], c_c, bottom=c+cc, color='g', width=0.5-((x_int[-1]-1)//7-5)*0.025)
plt.xticks(np.arange(0, (x_int[-1]-1)//7, 1), labelss, fontsize=15)
plt.yticks(fontsize=15)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.title('Alert amount', fontsize=23)

plt.subplot(5, 3, 15)
plt.ylim(int(np.min(std_MAESC))-1, int(np.max(std_MAESC))+1)
plt.plot([i for i in range(1, (x_int[-1]-1)//7+1)], std_con,  c='#00008B', lw=2, label='Condition')
plt.plot([i for i in range(1, (x_int[-1]-1)//7+1)], std_change, c='#800000', lw=2, label='Condition changes')
plt.scatter([i for i in range(1, (x_int[-1]-1)//7+1)], std_con, c = '#00008B', s=70, zorder=3)
plt.scatter([i for i in range(1, (x_int[-1]-1)//7+1)], std_change, c='#800000', s=70, zorder=3)
plt.axhline(0.95, c='#00FFFF', ls='dashed', label='critical for condition', zorder=1)
plt.axhline(1, c='#FF00FF', ls='dashed', label='critical for condition changes', zorder=1)
plt.plot([i for i in range(1, (x_int[-1]-1)//7+1)], [std_MAESC[j][0] for j in range((x_int[-1]-1)//7)], c='#3CB371', lw=2, label='Mood')
plt.plot([i for i in range(1, (x_int[-1]-1)//7+1)], [std_MAESC[j][3] for j in range((x_int[-1]-1)//7)], c='#7CFC00', lw=2, label='Symptoms')
plt.scatter([i for i in range(1, (x_int[-1]-1)//7+1)], [std_MAESC[j][0] for j in range((x_int[-1]-1)//7)], c='#3CB371', s=70, zorder=3)
plt.scatter([i for i in range(1, (x_int[-1]-1)//7+1)], [std_MAESC[j][3] for j in range((x_int[-1]-1)//7)], c='#7CFC00', s=70, zorder=3)
plt.axhline(1.5, c='#556B2F', ls='dashed', label='critical for mood', zorder=1)
plt.axhline(2, c='#00FF7F', ls='dashed', label='critical for symptoms', zorder=1)
plt.xticks(np.arange(1, (x_int[-1]-1)//7+1, 1), labelss, fontsize=15)
plt.yticks(fontsize=15)
plt.title('Std weekly', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend(loc='upper right')

plt.subplot(5, 3, 10)
plt.plot(x, cnd, c='#800080', lw=4)
plt.ylim(-2.5, 5.5)
plt.yticks(np.arange(-2, 6, 1.0), ['Awful', 'Very bad', 'Bad', 'Danger', 'OK', 'Good', 'Very good', 'Excellent'], fontsize=15)
plt.xlim(0, max(x_int)+1)
plt.xticks(np.arange(1, max(x_int)+2, round(x_int[-1]/15)), dates_pr[::round(x_int[-1]/15)], fontsize=15)
plt.scatter(x, cnd, c = cnd, cmap='nipy_spectral_r', norm=mcolors.Normalize(vmin=-2.5, vmax=10.7), s=70, zorder=2)
plt.axhline(1.5, c='r', ls='dashed', label='critical value')
plt.axhline(cu_m, c='g', ls='dashed', label='mean value')
plt.title('Condition updated', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.savefig('image.png')
plt.show()