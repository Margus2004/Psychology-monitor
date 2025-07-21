# MAESC check
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def file_reading(f):
    '''
    function for file reading

    Parameters
    f (txt file): file with observation
        including 5 observed parameters and the amount of days of observation
        in view of skipped entries

    Returns
    1) list of entries with skipped ones
    2) Data for MAESC parameters: Mood (M), Activity (A), Energy (E), Symptoms (S) and Mind Clearness (C)
    '''
    entries_amount = int(f.readline().split()[-1])
    entries = np.array([i for i in range(2, entries_amount+2)])/2
    skipped = int(f.readline().split()[-1])
    skipped_indexes = f.readline().split()[2:]
    for i in range(skipped):
        skipped_indexes[i] = int(skipped_indexes[i])
    entries = np.delete(entries, skipped_indexes)
    values = []
    line = f.readline().split()
    while line:
        values.append(line[1:])
        line = f.readline().split()
    for i in range(len(values)):
        for j in range(len(values[i])):
            values[i][j] = int(values[i][j])
    return [entries, values]


file = open("file.txt", "r")
file_read = file_reading(file)
file.close()
entries = file_read[0]
mood = np.array(file_read[1][0])
activity = np.array(file_read[1][1])
energy = np.array(file_read[1][2])
symptoms = np.array(file_read[1][3])
mind_clearness = np.array(file_read[1][4])

'''entries = np.array([i for i in range(2, 60)])/2  # input for (60-2)/2 = 29 days twice a day
# Mood (M), Activity (A), Energy (E), Symptoms (S) amd Mind Clearness (C) respectively (MAESC parameters)
mood = np.array(np.random.randint(1, high=11, size=58))  # random input due to the scale from 1 to 10
activity = np.array(np.random.randint(1, high=11, size=58))
energy = np.array(np.random.randint(1, high=11, size=58))
symptoms = np.array(np.random.randint(1, high=11, size=58))
mind_clearness = np.array(np.random.randint(1, high=11, size=58))'''


# Condition function 1. version
def condition(m, s, a):
    '''
    function of the present condition based on mood, activity and symptoms level of each entry

    Parameters
    m (float): mood
    s (float): symptoms
    a (float): activity

    Returns
    3: condition is Good
    2: condition is OK
    1: condition is in Danger zone
    0: condition is Bad
    '''
    if a >= 5.25 and ((8 <= m <= 10 and s <= 4.5) or (7 <= m < 8 and s <= 3.5) or (5.5 <= m < 7 and s <= 3)):
        return 3
    elif a >= 4.5 and ((8 <= m <= 10 and s <= 5.75) or (7 <= m < 8 and s <= 5.5) or (6 <= m < 7 and s <= 4.5) or (5 <= m < 6 and s <= 4) or (4 <= m < 5 and s <= 3.25)):
        return 2
    elif a >= 4 and ((8 <= m <= 10 and s <= 7) or (7 <= m < 8 and s <= 6.5) or (6 <= m < 7 and s <= 6) or (5 <= m < 6 and s <= 5.5) or (4 <= m < 5 and s <= 4.25)):
        return 1
    else:
        return 0


# Condition function 2. version
def condition_2(m, s, a):
    '''
    function of the present condition based on mood, activity and symptoms level of each entry,
    updated through the calculation of biases
    -> continuous function

    Parameters
    m (float): mood
    s (float): symptoms
    a (float): activity

    Returns
    condition, the bigger number represents the better condition
    '''
    condition_bias_interim = 0
    conditions = [[[8, 10.1, 4.5, 5.25], [7, 8, 3.5, 5.25], [5.5, 7, 3, 5.25]], [[8, 10.1, 5.75, 4.5], [7, 8, 5.5, 4.5], [6, 7, 4.5, 4.5], [5, 6, 4, 4.5], [4, 5, 3.25, 4.5]], [[8, 10.1, 7, 4], [7, 8, 6.5, 4], [6, 7, 6, 4], [5, 6, 5.5, 4], [4, 5, 4.25, 4]]]
    if condition(m, s, a) == 3:
        for c in conditions[0]:
            if c[0] <= m < c[1]:
                condition_bias_interim += 1.25 * (c[2] - s) + 0.8*(a - c[3])
                break
        condition_bias = condition_bias_interim/5.25
        return round(3+condition_bias, 3)
    elif 0 < condition(m, s, a) < 3:
        for c in conditions[3-condition(m, s, a)]:
            if c[0] <= m < c[1]:
                condition_bias_interim += 1.25*(c[2] - s) + 0.8*(a - c[3])
                break
        condition_bias = condition_bias_interim/5.25
        return round(condition(m, s, a)+condition_bias, 3)
    else:
        if conditions[2][-1][0] <= m < conditions[2][0][1]:
            for c in conditions[2]:
                if c[0] <= m < c[1]:
                    condition_bias_interim += 1.25*(c[2] - s) + 0.8*(a - c[3])
                    break
        else:
            condition_bias_interim += 2*(m - conditions[2][-1][0]) + 1.25*(conditions[2][-1][2] - s) + 0.8*(a - conditions[2][-1][3])
        condition_bias = condition_bias_interim/5.25
        if condition_bias >= 0:
            return round(1-condition_bias, 3)
        elif -1 < condition_bias < 0:
            return round(condition_bias+1, 3)
        else:
            return -round(-condition_bias-1, 3)


# Activity-energy function
def acten(a, e):
    '''
    function of Activity-Energy balance

    Parameters
    a (float): activity
    e (float): energy

    Returns
    -1: too low activity/too high energy
    1: too high activity/too low energy
    '''
    if a-0.5 <= e <= a+0.5:
        return 0
    elif e > a+1:
        return -1
    else:
        return 1


entries_int = np.array(entries, dtype=int)
day = len(np.unique(entries_int))  # day, not the entries amount


def date(d, m):
    '''
    dates list generator

    Parameters
    d (int): day when the observation starts
    m (int): month when the observation starts

    Returns
    dates_continuous: continuous interval of dates including skipped days
    date: dates of observation without skipped days
    '''
    dates_continuous = []
    calendar = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    l = np.setdiff1d(np.where(entries_int == entries, entries_int, 0), 0)
    del_dates = []
    for j in range(len(l) - 1):
        if (l[j + 1] - l[j]) > 1:
            del_dates.append([j, int(l[j + 1] - l[j])])
    for i in range(int(max(entries))):
        if m < 12:
            if d <= calendar.get(m):
                dates_continuous.append(str(d) + '/' + str(m))
                d += 1
            else:
                d = 1
                m += 1
                dates_continuous.append(str(d) + '/' + str(m))
                d += 1
        elif m == 12:
            if d <= calendar.get(m):
                dates_continuous.append(str(d) + '/' + str(m))
                d += 1
            else:
                d = 1
                m = 1
                dates_continuous.append(str(d) + '/' + str(m))
                d += 1
    dates_continuous.append(str(d) + '/' + str(m))
    dates_interim = dates_continuous.copy()
    for j in range(len(del_dates)):
        dates_interim = np.delete(dates_interim,[[i for i in range(del_dates[j][0] + 1, del_dates[j][0] + del_dates[j][1])]])
    dates = []
    for i in range(len(dates_interim)):
        dates.append(str(dates_interim[i]))  # dates adjusted for skipped days
    return [dates_continuous, dates]


dates_continuous = date(20, 10)[0]
dates = date(20, 10)[1]

# calculation of interim values of MAESC parameters
interim = []
maesc = np.array([mood, activity, energy, symptoms, mind_clearness])
for i in maesc:
        interim.append(np.where(entries_int==entries, i, 11))

condition_entries = np.array([])
for i in range(len(entries)):
    condition_entries = np.append(condition_entries, condition(mood[i], symptoms[i], activity[i]))

weghts_for_condition = [0.5 for j in range(len(entries))]
for i in range(len(entries)-1):
    if entries[i+1] != entries[i]+0.5 and (entries[i] % 2 == 0.0 or entries[i] % 2 == 1.0):
        weghts_for_condition[i] = 1
days_list = np.unique(entries_int)

#  mean values calculation
average = []
for i in maesc:
    average.append(np.average(mood, weights=weghts_for_condition))

# standard deviation calculation for Mood and Symptoms
mood_deviation = np.std(mood)
symptoms_deviation = np.std(symptoms)

condition_daily_interim = np.where(entries_int==entries, condition_entries, 5)  # mean condition daily with zeros (5)
condition_morning_interim = np.setdiff1d(np.where(entries_int==entries, condition_entries, 5), 5, assume_unique=True)  # condition values in the condition_morning
condition_evening = np.setdiff1d(np.where(entries_int!=entries, condition_entries, 5), 5, assume_unique=True)  # condition values in the evening

# mean condition and mean MAESC parameters daily calculation
for i in range(len(entries)):
    if condition_daily_interim[i] == 5:
        condition_daily_interim[i - 1] = condition((mood[i]+mood[i-1])/2, (symptoms[i]+symptoms[i-1])/2, (activity[i]+activity[i-1])/2)  # condition_daily_interim = condition for days with zeros (5)
        interim[0][i - 1] = (mood[i] + mood[i - 1]) / 2
        interim[1][i - 1] = (activity[i] + activity[i - 1]) / 2
        interim[2][i - 1] = (energy[i] + energy[i - 1]) / 2
        interim[3][i - 1] = (symptoms[i] + symptoms[i - 1]) / 2
        interim[4][i - 1] = (mind_clearness[i] + mind_clearness[i - 1]) / 2
condition_daily = np.setdiff1d(condition_daily_interim, 5, assume_unique=True)  # condition_daily = condition daily without zeros (5)
# MAESC daily values
mood_daily = np.setdiff1d(interim[0], 11, assume_unique=True)
activity_daily = np.setdiff1d(interim[1], 11, assume_unique=True)
energy_daily = np.setdiff1d(interim[2], 11, assume_unique=True)
symptoms_daily = np.setdiff1d(interim[3], 11, assume_unique=True)
mind_clearness_daily = np.setdiff1d(interim[4], 11, assume_unique=True)

condition_morning = np.copy(condition_morning_interim)  # calculation of true condition_morning condition values
deleted_items_morning_interim = []
i = 0
j = 0
k = 0
deleted_items_changes = []
while i < len(entries)-1:
    if entries[i+1] != entries[i]+0.5 and (entries[i] % 2 == 0.0 or entries[i] % 2 == 1.0):
        condition_morning = np.delete(condition_morning, j)
        deleted_items_morning_interim.append(i)
        deleted_items_changes.append(k)
        k += 1
    elif entries[i] % 2 == 0.5 or entries[i] % 2 == 1.5:
        j += 1
    else:
        k += 1
    i += 1
if entries[-1] % 2 == 0.0 or entries[-1] % 2 == 1.0:
    condition_morning = np.delete(condition_morning, -1)
    deleted_items_morning_interim.append(len(condition_morning)-1)
deleted_items_morning = np.append(deleted_items_morning_interim, len(dates)-1)

# condition changes calculation (condition_morning condition to evening condition)
change_0 = np.where(condition_morning == condition_evening, 0, 5)
change_1_n = np.where(condition_morning-1 == condition_evening, -1, 5)
change_2_n = np.where(condition_morning-2 == condition_evening, -2, 5)
change_3_n = np.where(condition_morning-3 == condition_evening, -3, 5)
change_1 = np.where(condition_morning+1 == condition_evening, 1, 5)
change_2 = np.where(condition_morning+2 == condition_evening, 2, 5)
change_3 = np.where(condition_morning+3 == condition_evening, 3, 5)
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

condition_average = np.average(condition_entries, weights=weghts_for_condition)  # weighted condition average/mean value (for entries)
condition_deviation = np.std(condition_entries)
condition_daily_average = np.mean(condition_daily)
condition_daily_deviation = np.std(condition_daily)
changes_average = np.mean(change)  # mean value for condition changes
changes_deviation = np.std(change)

entries_change = np.delete(days_list, deleted_items_changes)  # days when changes are possible to calculate
if entries[-1]%2==0.0 or entries[-1]%2==1.0:
    entries_change = np.delete(entries_change, -1)

condition_2_version = np.array([condition_2(mood[i], symptoms[i], activity[i]) for i in range(len(entries))])  # condition according to 2. version function

# bad, danger, ok and good days according to condition
bad_daily = np.where(condition_daily == 0, 1, 0)
danger_daily = np.where(condition_daily == 1, 1, 0)
ok_daily = np.where(condition_daily == 2, 1, 0)
good_daily = np.where(condition_daily == 3, 1, 0)

# bad, danger, ok and good entries according to condition
bad_entries = np.where(condition_entries == 0, 1, 0)
danger_entries = np.where(condition_entries == 1, 1, 0)
ok_entries = np.where(condition_entries == 2, 1, 0)
good_entries = np.where(condition_entries == 3, 1, 0)

# Activity-energy function into list of values
counter_acten = []
acten_data = []
for i in range(0, day):
    acten_data.append(acten(activity_daily[i], energy_daily[i]))


def choose(item, list):
    if item == 'mean':
        return float(np.mean(list))
    elif item == 'std':
        return float(np.std(list))
    else:
        return float(np.count_nonzero(list))
# weekly values
def weekly_values(data):
    for k in range(len(data)):
        for week in range((entries_int[-1] - 1) // 7):
            if len(data[k][1]) == 5:
                data[k][0].append([choose(data[k][3], data[k][1][j][np.where(data[k][2] >= (week * 7 + 1))[0][0]:np.where(data[k][2] <= (week + 1) * 7)[0][-1] + 1]) for j in range(len(data[k][1]))])
            else:
                if data[k][3] == 'average':
                    data[k][0].append(float(np.average(data[k][1][np.where(data[k][2] >= (week * 7 + 1))[0][0]:np.where(data[k][2] <= (week + 1) * 7)[0][-1] + 1], weights=weghts_for_condition[np.where(data[k][2]>=(week*7+1))[0][0]:np.where(data[k][2]<=(week+1)*7)[0][-1]+1])))
                else:
                    data[k][0].append(choose(data[k][3], data[k][1][np.where(data[k][2] >= (week * 7 + 1))[0][0]:np.where(data[k][2] <= (week + 1) * 7)[0][-1] + 1]))

    return None


mean_MAESC_week = []
deviation_MAESC_week = []
mean_condition_week = []
deviation_condition_week = []
mean_condition_2_version_week = []
deviation_condition_2_version_week = []
mean_condition_changes_week = []
deviation_condition_changes_week = []

values = [[mean_MAESC_week, maesc, entries_int, 'mean'], [mean_condition_week, condition_entries, entries_int, 'average'], [mean_condition_changes_week, change, entries_change, 'mean'], [deviation_MAESC_week, maesc, entries_int, 'std'], [deviation_condition_week, condition_entries, entries_int, 'std'], [deviation_condition_changes_week, change, entries_change, 'std'], [counter_acten, acten_data, np.unique(entries_int), 'count_nonzero'], [mean_condition_2_version_week, condition_2_version, entries_int, 'average'], [deviation_condition_2_version_week, condition_2_version, entries_int, 'std']]

condition_2_version_average = float(np.average(condition_2_version, weights=weghts_for_condition))

weekly_values(values)

'''for i in range((entries_int[-1]-1)//7):
    mean_MAESC_weekly.append([float(np.mean(mood[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.mean(activity[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.mean(energy[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.mean(symptoms[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.mean(mind_clearness[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1]))])
    mean_condition_weekly.append(float(np.average(condition_entries[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1], weights=weghts_for_condition[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])))
    mean_condition_changes_weekly.append(float(np.mean(change[np.where(entries_change>=(i*7+1))[0][0]:np.where(entries_change<=(i+1)*7)[0][-1]+1])))
    deviation_MAESC_weekly.append([float(np.std(mood[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.std(activity[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.std(energy[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.std(symptoms[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.std(mind_clearness[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1]))])
    deviation_condition_weekly.append(float(np.std(condition_entries[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])))
    deviation_condition_changes_weekly.append(float(np.std(change[np.where(entries_change>=(i*7+1))[0][0]:np.where(entries_change<=(i+1)*7)[0][-1]+1])))
    counter_acten.append(np.count_nonzero(acten_data[np.where(np.unique(entries_int)>=(i*7+1))[0][0]:np.where(np.unique(entries_int)<=(i+1)*7)[0][-1]+1]))
    mean_condition_2_version_weekly.append(float(np.average(condition_2_version[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1], weights=weghts_for_condition[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])))
    deviation_condition_2_version_weekly.append(float(np.std(condition_2_version[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])))
'''


def alert(x, l):
    '''
    function of alert detection

    Parameters
    x (): examined parameter
    l (): allowed values of the parameter
    red or yellow alert
    '''
    counter_red = 0
    counter_yellow = 0
    sign = int(np.sign(l[1]-l[0]))
    if sign*x < sign*l[0]:
        counter_red += 1
    elif sign*l[0] <= sign*x < sign*l[1]:
        counter_yellow += 1
    return [counter_red, counter_yellow]

list_param = [[] for week in range((entries_int[-1] - 1) // 7)]
# Alert calculation => week evaluation
red_alert = [0 for j in range((entries_int[-1]-1)//7)]
yellow_alert = [0 for j in range((entries_int[-1]-1)//7)]
list_values = [[5.35, 6], [5.25, 5.8], [5, 5.4], [5.8, 5.1], [5.8, 6.25], [1.5, 1.2], [2, 1.69], [1.5, 1.7], [0.95, 0.8], [1.5, 1.7], [0.95, 0.8], [1, 0.88], [3, 2], [0.5, 0.4]]
for week in range((entries_int[-1] - 1) // 7):
    for k in range(len(values)):
        if k == 0:
            for j in range(5):
                list_param[week].append(values[k][0][week][j])
        elif k == 3:
            list_param[week].append(values[k][0][week][0])
            list_param[week].append(values[k][0][week][3])
        else:
            list_param[week].append(values[k][0][week])
    for i in range(14):
        red_alert[week] += alert(list_param[week][i], list_values[i])[0]
        yellow_alert[week] += alert(list_param[week][i], list_values[i])[1]

red_alert = np.array(red_alert)  # red zone weekly
yellow_alert = np.array(yellow_alert)  # yellow zone weekly
green_zone = np.array(13-(red_alert+yellow_alert))  # green zone weekly

# Plots


def scatt(x, y, m, nor):
    plt.scatter(x, y, c=y, cmap = m, norm = nor, s=70, zorder=2)


def plotting(n, x, y, a, b, c, d, lab_x, lab_y, e, f, col_plot, cr_val, m_val, tit, line_w, lab_plot, mark, mark_s, amount, xlim):
    '''
    Plotting function (simple plots)

    Parameters:
    n (int): subplot number
    x (list or numpy array): x-axis values
    y (list or numpy array): y-axis values
    a (float): start value for y-axis
    b (float): end value for y-axis
    c (float): start value for ticks on y-axis
    d (float): end value for ticks on y-axis
    lab_x (string): labels for x-axis ticks
    lab_y (string): labels for y-axis ticks
    e (float): end value for ticks on x-axis
    f (int): tick spacing for x-axis
    col_plot (string): colour of the main plot line
    cr_val (float): critical value of the plot parameter
    m_val (float): mean value of the plot parameter
    tit (string): title of the subplot
    line_w (int or list of integers): width of plot line
    lab_plot (None, string or list of strings): labels of the plot line(s)
    mark (None or string): type of marker on the plot line
    mark_s (None or float): size of marker on the plot line
    amount (int): amount of plot lines in each subplot
    xlim (bool): if plt.xlim function is used in the subplot
    '''
    plt.subplot(5, 3, n)
    if xlim:
        plt.xlim(0, e+1)
    plt.xticks(np.arange(1, e+2, f), lab_x, fontsize=15)
    plt.ylim(a, b)
    plt.yticks(np.arange(c, d+1, 1.0), lab_y, fontsize=15)
    if amount == 1:
        plt.plot(x, y, c=col_plot, lw=line_w, label=lab_plot, marker=mark, ms=mark_s)
    else:
        for i in range(amount):
            plt.plot(x[i], y[i], c=col_plot[i], lw=line_w[i], label=lab_plot[i], marker=mark[i], ms=mark_s[i])
    if cr_val != 1:
        plt.axhline(cr_val, c='r', ls='dashed', label='critical value')
    if m_val != 500:
        plt.axhline(m_val, c='g', ls='dashed', label='mean value')
    plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
    plt.title(tit, fontsize=23)
    plt.legend(loc='upper left')
    return None


plt.figure(figsize=(52, 43), dpi=100)
plt.subplots_adjust(hspace=0.5, left=0.03, right=0.97, bottom=0.04, top=0.96)


nums = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15]
x_axis = [entries, entries, entries, entries, entries, entries, days_list, entries_change, entries, [[i for i in range(1, (entries_int[-1]-1)//7+1)] for j in range(2)], [[i for i in range(1, (entries_int[-1]-1)//7+1)] for j in range(5)], [[i for i in range(1, (entries_int[-1]-1)//7+1)] for j in range(4)]]
y_axis = [mood, activity, energy, symptoms, mind_clearness, condition_entries, condition_daily, change, condition_2_version, [mean_condition_week, mean_condition_changes_week], [[mean_MAESC_week[j][k] for j in range((entries_int[-1]-1)//7)] for k in range(5)], [deviation_condition_week, deviation_condition_changes_week, [deviation_MAESC_week[j][0] for j in range((entries_int[-1]-1)//7)], [deviation_MAESC_week[j][3] for j in range((entries_int[-1]-1)//7)]]]
x_lim_ticks = [[max(entries_int), round(entries_int[-1]/15)], [max(entries_int), round(entries_int[-1]/15)], [max(entries_int), round(entries_int[-1]/15)], [max(entries_int), round(entries_int[-1]/15)], [max(entries_int), round(entries_int[-1]/15)], [max(entries_int), round(entries_int[-1]/15)], [max(days_list), round(len(days_list)/15)], [max(days_list), round(len(days_list)/15)], [max(entries_int), round(entries_int[-1]/15)], [(entries_int[-1]-1)//7-1, 1], [(entries_int[-1]-1)//7-1, 1], [(entries_int[-1]-1)//7-1, 1]]
y_lim_ticks = [[0, 10, 0, 10], [0, 10, 0, 10], [0, 10, 0, 10], [0, 10, 0, 10], [0, 10, 0, 10], [-0.5, 3.5, 0, 3], [-0.5, 3.5, 0, 3], [-3.5, 3.5, -3, 3], [-2.5, 5.5, -2, 5], [int(min(mean_condition_changes_week))-1, int(max(mean_condition_week))+1, int(min(mean_condition_changes_week))-1, int(max(mean_condition_week))+1], [int(np.min(mean_MAESC_week)), int(np.max(mean_MAESC_week))+2, int(np.min(mean_MAESC_week)), int(np.max(mean_MAESC_week))+2], [int(np.min(deviation_MAESC_week))-1, int(np.max(deviation_MAESC_week))+1, int(np.min(deviation_MAESC_week))-1, int(np.max(deviation_MAESC_week))+1]]
colours = ['#3CB371', '#FFA500', '#FF1493', '#7CFC00', '#FFD700', '#00008B', '#00CED1', '#800000', '#800080', ['#00008B', '#800000'], ['#3CB371', '#FFA500', '#FF1493', '#7CFC00', '#800080'], ['#00008B', '#800000', '#3CB371', '#7CFC00']]
cr_values = [6, 5.8, 5.4, 5.8, 5.8, 1.5, 1.5, 1.5, 1.5, 1, 1, 1]
av_values = [average[0], average[1], average[2], average[3], average[4], condition_average, condition_daily_average, changes_average, condition_2_version_average, 500, 500, 500, 500]
titles = ['Mood', 'Activity', 'Energy', 'Symptoms', 'Mind clearness', 'Condition', 'Condition daily', 'Condition changes', 'Condition updated', 'Mean weekly Condition', 'Mean weakly MAESC', 'Deviation weekly']
sc = [False, False, False, False, False, True, True, True, True, False, False, False]
maps = [mcolors.ListedColormap(['r', '#FF8C00', '#FFD700', 'g']), mcolors.ListedColormap(['r', '#FF8C00', '#FFD700', 'g']), mcolors.ListedColormap(['#0000CD', '#1E90FF', '#00FFFF', 'g', '#FFA500', '#FF4500', '#B22222']), 'nipy_spectral_r']
norms = [None, None, None, mcolors.Normalize(vmin=-2.5, vmax=10.7)]
labelss = [dates_continuous[::7][i]+'-'+dates_continuous[6::7][i] for i in range((entries_int[-1]-1)//7)]
labels_x = [dates_continuous[::round(len(days_list)/15)], dates_continuous[::round(len(days_list)/15)], dates_continuous[::round(len(days_list)/15)], dates_continuous[::round(len(days_list)/15)], dates_continuous[::round(len(days_list)/15)], dates_continuous[::round(len(days_list)/15)], dates_continuous[::round(len(days_list)/15)], dates_continuous[::round(len(days_list)/15)], dates_continuous[::round(len(days_list)/15)], labelss, labelss, labelss]
labels_y = [None, None, None, None, None, ['Bad', 'Danger', 'OK', 'Good'], ['Bad', 'Danger', 'OK', 'Good'], None, ['Awful', 'Very bad', 'Bad', 'Danger', 'OK', 'Good', 'Very good', 'Excellent'], None, None, None]
line_width = [4, 4, 4, 4, 4, 4, 4, 4, 4, [2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2]]
labels_plot = [None, None, None, None, None, None, None, None, None, ['Condition', 'Condition changes'], ['Mood', 'Activity', 'Energy', 'Symptoms', 'Mind clearness'], ['Condition', 'Condition changes', 'Mood', 'Symptoms']]
markers = [None, None, None, None, None, None, None, None, None, ['.', '.'], ['.', '.', '.', '.', '.'], ['.', '.', '.', '.']]
markers_size = [None, None, None, None, None, None, None, None, None, [18, 18], [18, 18, 18, 18, 18], [18, 18, 18, 18]]
amounts = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 5, 4]
xlims = [True, True, True, True, True, True, True, True, True, False, False, False]

j = 0
for i in range(len(nums)):
    plotting(nums[i], x_axis[i], y_axis[i], y_lim_ticks[i][0], y_lim_ticks[i][1], y_lim_ticks[i][2], y_lim_ticks[i][3], labels_x[i], labels_y[i], x_lim_ticks[i][0], x_lim_ticks[i][1], colours[i], cr_values[i], av_values[i], titles[i], line_width[i], labels_plot[i], markers[i], markers_size[i], amounts[i], xlims[i])
    if nums[i] == 9:
        plt.axhline(-1.5, c='r', ls='dashed', label='critical value')
    if sc[i]:
        scatt(x_axis[i], y_axis[i], maps[j], norms[j])
        j += 1


def plotting_pie(n, labls, cols, plott, radus, pct_dist, width, amount, tit):
    '''
    plotting function (pie plots)

    Parameters
    n (int): subplot number
    labls (list of strings or None): labels for pie chart
    cols (list of strings): colours of pie chart
    plott (numpy array of floats): plotting content
    radus (float): radius of the pie chart
    pct_dist (float): distance of the percentage (from the middle to the edge) on the pie chart
    width (float): width of the ring (pie chart)
    amount (int): amount of rings int the pie chart
    tit (string): title of the chart
    '''
    plt.subplot(5, 3, n)
    for i in range(amount):
        plt.pie(plott[i], radius=radus[i], labels=labls[i], colors=cols, startangle = 180, autopct='%1.1f%%', pctdistance=pct_dist[i], textprops={'fontsize': 14}, wedgeprops=dict(width=width, edgecolor='w'))
    plt.title(tit, fontsize=23)
    return None


nums = [11, 12]
labs = [[['Bad', 'Danger', 'OK', 'Good'], None], [['-3', '-2', '-1', '0', '1', '2', '3']]]
colours = [['r', '#FF8C00', '#FFD700', 'g'], ['#0000CD', '#1E90FF', '#00FFFF', 'g', '#FFA500', '#FF4500', '#B22222']]
plots = [[np.array([np.count_nonzero(bad_daily)/day, np.count_nonzero(danger_daily)/day, np.count_nonzero(ok_daily)/day, np.count_nonzero(good_daily)/day]), np.array([np.count_nonzero(bad_entries)/len(entries), np.count_nonzero(danger_entries)/len(entries), np.count_nonzero(ok_entries)/len(entries), np.count_nonzero(good_entries)/len(entries)])], [np.array([count_neg_3, count_neg_2, count_neg_1, count_zero, count_pos_1, count_pos_2, count_pos_3])]]
rads = [[1, 0.55], [1]]
dists = [[0.8, 0.6], [0.6]]
widths = [0.4, 0.85]
ams = [2, 1]
tits = ['Condition percentage', 'Condition changes']

for i in range(2):
    plotting_pie(nums[i], labs[i], colours[i], plots[i], rads[i], dists[i], widths[i], ams[i], tits[i])


def plotting_bar(zone, bott, col):
    '''
    plotting function (bar plot)

    Parameters
    zone (numpy array): parts of column of the bar plot
    bott (numpy array): the bottom of the given zone
    col (string): colour of the zone
    '''
    plt.subplot(5, 3, 6)
    for i in range(3):
        plt.bar([i for i in range((entries_int[-1]-1)//7)], zone[i], bottom=bott[i], color=col[i], width=0.5-((entries_int[-1]-1)//7-5)*0.025)
    plt.ylim(0, 13)
    plt.xticks(np.arange(0, (entries_int[-1] - 1) // 7, 1), [dates_continuous[::7][i]+'-'+dates_continuous[6::7][i] for i in range((entries_int[-1]-1)//7)], fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(color='#191970', linestyle='dashed', linewidth=0.5)
    plt.title('Alert amount', fontsize=23)
    return None


plots = [red_alert, yellow_alert, green_zone]
botts = [None, red_alert, red_alert+yellow_alert]
cols = ['r', '#FF8C00', 'g']

plotting_bar(plots, botts, cols)


plt.savefig('image.png')
plt.show()