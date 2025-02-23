# MAESC check
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

entries = np.array([i for i in range(2, 60)])/2  # input for (60-2)/2 = 29 days twice a day
# Mood (M), Activity (A), Energy (E), Symptoms (S) amd Mind Clearness (C) respectively (MAESC parameters)
mood = np.array(np.random.randint(1, high=11, size=58))  # random input due to the scale from 1 to 10
activity = np.array(np.random.randint(1, high=11, size=58))
energy = np.array(np.random.randint(1, high=11, size=58))
symptoms = np.array(np.random.randint(1, high=11, size=58))
mind_clearness = np.array(np.random.randint(1, high=11, size=58))


# Condition function 1. version
def condition(m, s, a):
    '''
    
    :param m: mood
    :param s: symptoms
    :param a: activity
    :return: 3:=Good, 2:=OK, 1:=Danger, 0:=Bad
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
def condition_uc(m, s, a):
    '''
    
    :param m: mood
    :param s: symptoms
    :param a: activity
    :return: condition weighted and continuous
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
    
    :param a: activity
    :param e: energy
    :return: -1:=low activity/high energy, 1:=high activity/low energy
    '''
    if a-0.5 <= e <= a+0.5:
        return 0
    elif e > a+1:
        return -1
    else:
        return 1


entries_int = np.array(entries, dtype=int)
day = len(np.unique(entries_int))  # day, not the entries` amount

# date list generator
dates_continuous = []
calendar = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
d = 20  # the day of the 1. entry
m = 10  # the month of the 1. entry
l = np.setdiff1d(np.where(entries_int==entries, entries_int, 0), 0)
del_dates = []
for j in range(len(l)-1):
    if (l[j+1]-l[j]) > 1:
        del_dates.append([j, int(l[j+1]-l[j])])
for i in range(int(max(entries))):
    if m < 12:
        if d <= calendar.get(m):
            dates_continuous.append(str(d)+'/'+str(m))
            d += 1
        else:
            d = 1
            m += 1
            dates_continuous.append(str(d) + '/' + str(m))
            d += 1
    elif m == 12:
        if d <= calendar.get(m):
            dates_continuous.append(str(d)+'/'+str(m))
            d += 1
        else:
            d = 1
            m = 1
            dates_continuous.append(str(d) + '/' + str(m))
            d += 1
dates_continuous.append(str(d) + '/' + str(m))

dates_interim = dates_continuous.copy()
for j in range(len(del_dates)):
    dates_interim = np.delete(dates_interim, [[i for i in range(del_dates[j][0]+1, del_dates[j][0]+del_dates[j][1])]])

dates = []
for i in range(len(dates_interim)):
    dates.append(str(dates_interim[i]))  # dates adjusted for skipped days

mood_interim = np.where(entries_int==entries, mood, 11)
activity_interim = np.where(entries_int==entries, activity, 11)
energy_interim = np.where(entries_int==entries, energy, 11)
symptoms_interim = np.where(entries_int==entries, symptoms, 11)
mind_clearness_interim = np.where(entries_int==entries, mind_clearness, 11)
condition_entries = np.array([])
for i in range(len(entries)):
    condition_entries = np.append(condition_entries, condition(mood[i], symptoms[i], activity[i]))

weghts_for_condition = [0.5 for j in range(len(entries))]
for i in range(len(entries)-1):
    if entries[i+1] != entries[i]+0.5 and (entries[i] % 2 == 0.0 or entries[i] % 2 == 1.0):
        weghts_for_condition[i] = 1
days_list = np.unique(entries_int)

#  mean values calculation
mood_average = np.average(mood, weights=weghts_for_condition)
activity_average = np.average(activity, weights=weghts_for_condition)
energy_average = np.average(energy, weights=weghts_for_condition)
symptoms_average = np.average(symptoms, weights=weghts_for_condition)
mind_clearness_average = np.average(mind_clearness, weights=weghts_for_condition)
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
        mood_interim[i - 1] = (mood[i] + mood[i - 1]) / 2
        activity_interim[i - 1] = (activity[i] + activity[i - 1]) / 2
        energy_interim[i - 1] = (energy[i] + energy[i - 1]) / 2
        symptoms_interim[i - 1] = (symptoms[i] + symptoms[i - 1]) / 2
        mind_clearness_interim[i - 1] = (mind_clearness[i] + mind_clearness[i - 1]) / 2
condition_daily = np.setdiff1d(condition_daily_interim, 5, assume_unique=True)  # condition_daily = condition daily without zeros (5)
# MAESC daily values
mood_daily = np.setdiff1d(mood_interim, 11, assume_unique=True)
activity_daily = np.setdiff1d(activity_interim, 11, assume_unique=True)
energy_daily = np.setdiff1d(energy_interim, 11, assume_unique=True)
symptoms_daily = np.setdiff1d(symptoms_interim, 11, assume_unique=True)
mind_clearness_daily = np.setdiff1d(mind_clearness_interim, 11, assume_unique=True)

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

condition_2_version = np.array([condition_uc(mood[i], symptoms[i], activity[i]) for i in range(len(entries))])  # condition according to 2. version function

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

# weekly values
mean_MAESC_weekly = []
deviation_MAESC_weekly = []
mean_condition_weekly = []
deviation_condition_weekly = []
mean_condition_2_version_weekly = []
deviation_condition_2_version_weekly = []
mean_condition_changes_weekly = []
deviation_condition_changes_weekly = []
condition_2_version_average = float(np.average(condition_2_version, weights=weghts_for_condition))

for i in range((entries_int[-1]-1)//7):
    mean_MAESC_weekly.append([float(np.mean(mood[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.mean(activity[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.mean(energy[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.mean(symptoms[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.mean(mind_clearness[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1]))])
    mean_condition_weekly.append(float(np.average(condition_entries[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1], weights=weghts_for_condition[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])))
    mean_condition_changes_weekly.append(float(np.mean(change[np.where(entries_change>=(i*7+1))[0][0]:np.where(entries_change<=(i+1)*7)[0][-1]+1])))
    deviation_MAESC_weekly.append([float(np.std(mood[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.std(activity[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.std(energy[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.std(symptoms[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])), float(np.std(mind_clearness[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1]))])
    deviation_condition_weekly.append(float(np.std(condition_entries[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])))
    deviation_condition_changes_weekly.append(float(np.std(change[np.where(entries_change>=(i*7+1))[0][0]:np.where(entries_change<=(i+1)*7)[0][-1]+1])))
    counter_acten.append(np.count_nonzero(acten_data[np.where(np.unique(entries_int)>=(i*7+1))[0][0]:np.where(np.unique(entries_int)<=(i+1)*7)[0][-1]+1]))
    mean_condition_2_version_weekly.append(float(np.average(condition_2_version[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1], weights=weghts_for_condition[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])))
    deviation_condition_2_version_weekly.append(float(np.std(condition_2_version[np.where(entries_int>=(i*7+1))[0][0]:np.where(entries_int<=(i+1)*7)[0][-1]+1])))


def alert(x, l):
    '''

    :param x: examined parameter
    :param l: allowed values of the parameter
    :return: red or yellow alert
    '''
    counter_red = 0
    counter_yellow = 0
    sign = np.sign(l[1]-l[0])
    if sign*x < sign*l[0]:
        counter_red += 1
    elif sign*l[0] <= sign*x < sign*l[1]:
        counter_yellow += 1
    return [counter_red, counter_yellow]

# Alert calculation -> week evaluation
red_alert = [0 for j in range((entries_int[-1]-1)//7)]
yellow_alert = [0 for j in range((entries_int[-1]-1)//7)]
for week in range((entries_int[-1]-1)//7):
    list_values = [[5.35, 6], [5.25, 5.8], [5, 5.4], [5.8, 5.1], [5.8, 6.25], [1.5, 1.2], [2, 1.69], [1.5, 1.7], [0.95, 0.8], [1.5, 1.7], [0.95, 0.8], [1, 0.88], [3, 2], [0.5, 0.4]]
    list_param = [mean_MAESC_weekly[week][0], mean_MAESC_weekly[week][1], mean_MAESC_weekly[week][2], mean_MAESC_weekly[week][3], mean_MAESC_weekly[week][4], deviation_MAESC_weekly[week][0], deviation_MAESC_weekly[week][3], mean_condition_weekly[week], deviation_condition_weekly[week], mean_condition_2_version_weekly[week], deviation_condition_2_version_weekly[week], deviation_condition_changes_weekly[week], counter_acten[week], np.abs(mean_condition_changes_weekly[week])]
    for j in range(14):
        red_alert[week] += alert(list_param[j], list_values[j])[0]
        yellow_alert[week] += alert(list_param[j], list_values[j])[1]

red_alert = np.array(red_alert)  # red zone weekly
yellow_alert = np.array(yellow_alert)  # yellow zone weekly
green_zone = np.array(13-(red_alert+yellow_alert))  # green zone weekly

# Plots
plt.figure(figsize=(52, 43), dpi=100)
plt.subplots_adjust(hspace=0.5, left=0.03, right=0.97, bottom=0.04, top=0.96)

plt.subplot(5, 3, 1)
plt.ylim(0, 10)
plt.xlim(0, max(entries_int)+1)
plt.xticks(np.arange(1, max(entries_int)+2, round(entries_int[-1]/15)), dates_continuous[::round(entries_int[-1]/15)], fontsize=15)
plt.yticks(np.arange(0, 11, 1.0), fontsize=15)
plt.plot(entries, mood, c='#3CB371', lw=4)
plt.axhline(6, c='r', ls='dashed', label='critical value')
plt.axhline(mood_average, c='g', ls='dashed', label='mean value')
plt.title('Mood', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 2)
plt.ylim(0, 10)
plt.xlim(0, max(entries_int)+1)
plt.xticks(np.arange(1, max(entries_int)+2, round(entries_int[-1]/15)), dates_continuous[::round(entries_int[-1]/15)], fontsize=15)
plt.yticks(np.arange(0, 11, 1.0), fontsize=15)
plt.plot(entries, activity, c='#FFA500', lw=4)
plt.axhline(5.8, c='r', ls='dashed', label='critical value')
plt.axhline(activity_average, c='g', ls='dashed', label='mean value')
plt.title('Activity', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 3)
plt.ylim(0, 10)
plt.xlim(0, max(entries_int)+1)
plt.xticks(np.arange(1, max(entries_int)+2, round(entries_int[-1]/15)), dates_continuous[::round(entries_int[-1]/15)], fontsize=15)
plt.yticks(np.arange(0, 11, 1.0), fontsize=15)
plt.plot(entries, energy, c='#FF1493', lw=4)
plt.axhline(5.4, c='r', ls='dashed', label='critical value')
plt.axhline(energy_average, c='g', ls='dashed', label='mean value')
plt.title('Energy', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 4)
plt.ylim(0, 10)
plt.xlim(0, max(entries_int)+1)
plt.xticks(np.arange(1, max(entries_int)+2, round(entries_int[-1]/15)), dates_continuous[::round(entries_int[-1]/15)], fontsize=15)
plt.yticks(np.arange(0, 11, 1.0), fontsize=15)
plt.plot(entries, symptoms, c='#7CFC00', lw=4)
plt.axhline(5.8, c='r', ls='dashed', label='critical value')
plt.axhline(symptoms_average, c='g', ls='dashed', label='mean value')
plt.title('Symptoms', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 5)
plt.ylim(0, 10)
plt.xlim(0, max(entries_int)+1)
plt.xticks(np.arange(1, max(entries_int)+2, round(entries_int[-1]/15)), dates_continuous[::round(entries_int[-1]/15)], fontsize=15)
plt.yticks(np.arange(0, 11, 1.0), fontsize=15)
plt.plot(entries, mind_clearness, c='#FFD700', lw=4)
plt.axhline(5.8, c='r', ls='dashed', label='critical value')
plt.axhline(mind_clearness_average, c='g', ls='dashed', label='mean value')
plt.title('Mind clearness', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 9)
plt.ylim(-3.5, 3.5)
plt.xlim(0, max(days_list)+1)
plt.xticks(np.arange(1, max(days_list)+2, round(len(days_list)/15)), dates_continuous[::round(len(days_list)/15)], fontsize=15)
plt.yticks(fontsize=15)
plt.scatter(entries_change, change, c = change, cmap = mcolors.ListedColormap(['#0000CD', '#1E90FF', '#00FFFF', 'g', '#FFA500', '#FF4500', '#B22222']), s=80, zorder=3)
plt.plot(entries_change, change, c='#800000', lw=4)
plt.axhline(1.5, c='r', ls='dashed', label='critical values')
plt.axhline(-1.5, c='r', ls='dashed')
plt.axhline(changes_average, c='g', ls='dashed', label='mean value')
plt.title('Condition changes', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 7)
plt.ylim(-0.5, 3.5)
plt.yticks(np.arange(0.0, 4.0, 1.0), ['Bad', 'Danger', 'OK', 'Good'], fontsize=15)
plt.xlim(0, max(entries_int)+1)
plt.xticks(np.arange(1, max(entries_int)+2, round(entries_int[-1]/15)), dates_continuous[::round(entries_int[-1]/15)], fontsize=15)
plt.plot(entries, condition_entries, c='#00008B', lw=4)
plt.scatter(entries, condition_entries, c = condition_entries, cmap = mcolors.ListedColormap(['r', '#FF8C00', '#FFD700', 'g']), s=70, zorder=2)
plt.axhline(1.5, c='r', ls='dashed', label='critical value')
plt.axhline(condition_average, c='g', ls='dashed', label='mean value')
plt.title('Condition', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 8)
plt.ylim(-0.5, 3.5)
plt.yticks(np.arange(0.0, 4.0, 1.0), ['Bad', 'Danger', 'OK', 'Good'], fontsize=15)
plt.xlim(0, max(days_list)+1)
plt.xticks(np.arange(1, max(days_list)+2, round(len(days_list)/15)), dates_continuous[::round(len(days_list)/15)], fontsize=15)
plt.plot(days_list, condition_daily, c='#00CED1', lw=4)
plt.scatter(days_list, condition_daily, c = condition_daily, cmap = mcolors.ListedColormap(['r', '#FF8C00', '#FFD700', 'g']), s=80, zorder=2)
plt.axhline(1.5, c='r', ls='dashed', label='critical value')
plt.axhline(condition_daily_average, c='g', ls='dashed', label='mean value')
plt.title('Condition', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 11)
labels = np.array(['Bad', 'Danger', 'OK', 'Good'])
yyyy = np.array([np.count_nonzero(bad_daily)/day, np.count_nonzero(danger_daily)/day, np.count_nonzero(ok_daily)/day, np.count_nonzero(good_daily)/day])
yyyyy = np.array([np.count_nonzero(bad_entries)/len(entries), np.count_nonzero(danger_entries)/len(entries), np.count_nonzero(ok_entries)/len(entries), np.count_nonzero(good_entries)/len(entries)])
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

labelss = [dates_continuous[::7][i]+'-'+dates_continuous[6::7][i] for i in range((entries_int[-1]-1)//7)]

plt.subplot(5, 3, 14)
plt.ylim(int(np.min(mean_MAESC_weekly)), int(np.max(mean_MAESC_weekly))+2)
plt.xlim(0.8, (entries_int[-1]-1)//7+0.2)
plt.xticks(np.arange(1, (entries_int[-1]-1)//7+1, 1), labelss, fontsize=15)
plt.yticks(fontsize=15)
plt.plot([i for i in range(1, (entries_int[-1]-1)//7+1)], [mean_MAESC_weekly[j][0] for j in range((entries_int[-1]-1)//7)], c='#3CB371', lw=2, label='Mood')
plt.plot([i for i in range(1, (entries_int[-1]-1)//7+1)], [mean_MAESC_weekly[j][1] for j in range((entries_int[-1]-1)//7)], c='#FFA500', lw=2, label='Activity')
plt.plot([i for i in range(1, (entries_int[-1]-1)//7+1)], [mean_MAESC_weekly[j][2] for j in range((entries_int[-1]-1)//7)], c='#FF1493', lw=2, label='Energy')
plt.plot([i for i in range(1, (entries_int[-1]-1)//7+1)], [mean_MAESC_weekly[j][3] for j in range((entries_int[-1]-1)//7)], c='#7CFC00', lw=2, label='Symptoms')
plt.plot([i for i in range(1, (entries_int[-1]-1)//7+1)], [mean_MAESC_weekly[j][4] for j in range((entries_int[-1]-1)//7)], c='#800080', lw=2, label='Mind Clearness')
plt.scatter([i for i in range(1, (entries_int[-1]-1)//7+1)], [mean_MAESC_weekly[j][0] for j in range((entries_int[-1]-1)//7)], c = '#3CB371', s=70, zorder=6)
plt.scatter([i for i in range(1, (entries_int[-1]-1)//7+1)], [mean_MAESC_weekly[j][1] for j in range((entries_int[-1]-1)//7)], c='#FFA500', s=70, zorder=6)
plt.scatter([i for i in range(1, (entries_int[-1]-1)//7+1)], [mean_MAESC_weekly[j][2] for j in range((entries_int[-1]-1)//7)], c='#FF1493', s=70, zorder=6)
plt.scatter([i for i in range(1, (entries_int[-1]-1)//7+1)], [mean_MAESC_weekly[j][3] for j in range((entries_int[-1]-1)//7)], c='#7CFC00', s=70, zorder=6)
plt.scatter([i for i in range(1, (entries_int[-1]-1)//7+1)], [mean_MAESC_weekly[j][4] for j in range((entries_int[-1]-1)//7)], c='#800080', s=70, zorder=6)
plt.title('Mean weakly MAESC', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 13)
plt.ylim(int(min(mean_condition_changes_weekly))-1, int(max(mean_condition_weekly))+1)
plt.plot([i for i in range(1, (entries_int[-1]-1)//7+1)], mean_condition_weekly,  c='#00008B', lw=2, label='Condition')
plt.plot([i for i in range(1, (entries_int[-1]-1)//7+1)], mean_condition_changes_weekly, c='#800000', lw=2, label='Condition changes')
plt.scatter([i for i in range(1, (entries_int[-1]-1)//7+1)], mean_condition_weekly, c = '#00008B', s=70, zorder=3)
plt.scatter([i for i in range(1, (entries_int[-1]-1)//7+1)], mean_condition_changes_weekly, c='#800000', s=70, zorder=3)
plt.xticks(np.arange(1, (entries_int[-1]-1)//7+1, 1), labelss, fontsize=15)
plt.yticks(fontsize=15)
plt.title('Mean weekly Condition', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.subplot(5, 3, 6)
plt.ylim(0, 13)
plt.bar([i for i in range((entries_int[-1]-1)//7)], red_alert, color='r', width=0.5-((entries_int[-1]-1)//7-5)*0.025)
plt.bar([i for i in range((entries_int[-1]-1)//7)], yellow_alert, bottom=red_alert, color='#FF8C00', width=0.5-((entries_int[-1]-1)//7-5)*0.025)
plt.bar([i for i in range((entries_int[-1]-1)//7)], green_zone, bottom=red_alert+yellow_alert, color='g', width=0.5-((entries_int[-1]-1)//7-5)*0.025)
plt.xticks(np.arange(0, (entries_int[-1]-1)//7, 1), labelss, fontsize=15)
plt.yticks(fontsize=15)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.title('Alert amount', fontsize=23)

plt.subplot(5, 3, 15)
plt.ylim(int(np.min(deviation_MAESC_weekly))-1, int(np.max(deviation_MAESC_weekly))+1)
plt.plot([i for i in range(1, (entries_int[-1]-1)//7+1)], deviation_condition_weekly,  c='#00008B', lw=2, label='Condition')
plt.plot([i for i in range(1, (entries_int[-1]-1)//7+1)], deviation_condition_changes_weekly, c='#800000', lw=2, label='Condition changes')
plt.scatter([i for i in range(1, (entries_int[-1]-1)//7+1)], deviation_condition_weekly, c = '#00008B', s=70, zorder=3)
plt.scatter([i for i in range(1, (entries_int[-1]-1)//7+1)], deviation_condition_changes_weekly, c='#800000', s=70, zorder=3)
plt.axhline(0.95, c='#00FFFF', ls='dashed', label='critical for condition', zorder=1)
plt.axhline(1, c='#FF00FF', ls='dashed', label='critical for condition changes', zorder=1)
plt.plot([i for i in range(1, (entries_int[-1]-1)//7+1)], [deviation_MAESC_weekly[j][0] for j in range((entries_int[-1]-1)//7)], c='#3CB371', lw=2, label='Mood')
plt.plot([i for i in range(1, (entries_int[-1]-1)//7+1)], [deviation_MAESC_weekly[j][3] for j in range((entries_int[-1]-1)//7)], c='#7CFC00', lw=2, label='Symptoms')
plt.scatter([i for i in range(1, (entries_int[-1]-1)//7+1)], [deviation_MAESC_weekly[j][0] for j in range((entries_int[-1]-1)//7)], c='#3CB371', s=70, zorder=3)
plt.scatter([i for i in range(1, (entries_int[-1]-1)//7+1)], [deviation_MAESC_weekly[j][3] for j in range((entries_int[-1]-1)//7)], c='#7CFC00', s=70, zorder=3)
plt.axhline(1.5, c='#556B2F', ls='dashed', label='critical for mood', zorder=1)
plt.axhline(2, c='#00FF7F', ls='dashed', label='critical for symptoms', zorder=1)
plt.xticks(np.arange(1, (entries_int[-1]-1)//7+1, 1), labelss, fontsize=15)
plt.yticks(fontsize=15)
plt.title('Std weekly', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend(loc='upper right')

plt.subplot(5, 3, 10)
plt.plot(entries, condition_2_version, c='#800080', lw=4)
plt.ylim(-2.5, 5.5)
plt.yticks(np.arange(-2, 6, 1.0), ['Awful', 'Very bad', 'Bad', 'Danger', 'OK', 'Good', 'Very good', 'Eentriescellent'], fontsize=15)
plt.xlim(0, max(entries_int)+1)
plt.xticks(np.arange(1, max(entries_int)+2, round(entries_int[-1]/15)), dates_continuous[::round(entries_int[-1]/15)], fontsize=15)
plt.scatter(entries, condition_2_version, c = condition_2_version, cmap='nipy_spectral_r', norm=mcolors.Normalize(vmin=-2.5, vmax=10.7), s=70, zorder=2)
plt.axhline(1.5, c='r', ls='dashed', label='critical value')
plt.axhline(condition_2_version_average, c='g', ls='dashed', label='mean value')
plt.title('Condition updated', fontsize=23)
plt.grid(color = '#191970', linestyle = 'dashed', linewidth = 0.5)
plt.legend()

plt.savefig('image.png')
plt.show()