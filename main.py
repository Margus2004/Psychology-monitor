# MAESC check
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def condition(m, s, a):
    '''
    General condition function dependent on mood, symptoms and activity
    The result is with appropriate coefficients:
    Condition = A*m - B*s + C*a, where A > B > C,
    and is normalized to [-3,5]:
    Condition = (A*m - B*s + C*a - D)/E, where D = (A+B+C)*10*3/8 - B*10, and E = (A+B+C)*10/8,
    as m, s, a are in [0, 10].

    Parameters
    m (float): mood
    s (float): symptoms
    a (float): activity

    Return
    general condition of the examined person (float in range [-3,5])
    Further:
    [-3,-2): Awful, [-2,-1): Very bad, [-1,0): Bad, [0,1): Danger zone,
    [1, 2): OK, [2, 3): Good, [3, 4): Very good, [4,5]: Excellent
    '''
    return (m - 0.6*s + 0.25*a - (18.5*3/8-6))/(18.5/8)


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


def daily(n, func, val):
    '''
    Daily condition identification function

    Parameters
    n (int): input amount
    func (array): function values
    val (2-dim. list): critical values for each function value

    Return
    daily_param (list): identification if the function value is in some range
    '''
    daily_param = []
    for i in range(n):
        daily_param.append(np.where((val[i][0] <= func) & (func < val[i][1]), 1, 0))
    return daily_param


def choose(item, lt):
    '''
    Function choice

    Parameters
    item (str): the choice parameter
    lt (list): examined list

    Return
    calculated mean value, standard deviation or amount of nonzero values
    '''
    if item == 'mean':
        return float(np.mean(lt))
    elif item == 'std':
        return float(np.std(lt))
    else:
        return float(np.count_nonzero(lt))


# weekly values
def weekly_values(data):
    '''
    Calculation of weekly statistics

    Parameters
    data (2- or 3-dim. array): data for the calculation

    Return
    None
    '''
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


def alert(x, l):
    '''
    Function of alert detection

    Parameters
    x (float): examined parameter
    l (list): allowed values of the parameter

    Return
    Red or yellow alert
    '''
    counter_red = 0
    counter_yellow = 0
    sign = int(np.sign(l[1]-l[0]))
    if sign*x < sign*l[0]:
        counter_red += 1
    elif sign*l[0] <= sign*x < sign*l[1]:
        counter_yellow += 1
    return [counter_red, counter_yellow]


# Plots

def scatt(x, y, m, nor):
    '''
    Function of scatter plotting

    x (list or numpy array): x-axis values
    y (list or numpy array): y-axis values
    m (Colormap): value for the cmap
    nor (mcolors.Normalize): value for the norm

    Return
    None
    '''
    plt.scatter(x, y, c=y, cmap = m, norm = nor, s=70, zorder=2)
    return None


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

    Return
    None
    '''
    plt.subplot(4, 3, n)
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


def plotting_pie(n, labls, cols, plott, tit):
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

    Return
    None
    '''
    plt.subplot(4, 3, n)
    plt.pie(plott, radius=1, labels=labls, colors=cols, startangle = 180, autopct='%1.1f%%', pctdistance=0.6, textprops={'fontsize': 14}, wedgeprops=dict(width=0.85, edgecolor='w'))
    plt.title(tit, fontsize=23)
    return None


def plotting_bar(zone, bott, col):
    '''
    plotting function (bar plot)

    Parameters
    zone (numpy array): parts of column of the bar plot
    bott (numpy array): the bottom of the given zone
    col (string): colour of the zone
    '''
    plt.subplot(4, 3, 6)
    for i in range(3):
        plt.bar([i for i in range((entries_int[-1]-1)//7)], zone[i], bottom=bott[i], color=col[i], width=0.5-((entries_int[-1]-1)//7-5)*0.025)
    plt.ylim(0, 12)
    plt.xticks(np.arange(0, (entries_int[-1] - 1) // 7, 1), [dates[::7][i]+'-'+dates[6::7][i] for i in range((entries_int[-1]-1)//7)], fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(color='#191970', linestyle='dashed', linewidth=0.5)
    plt.title('Alert amount', fontsize=23)
    return None


input_dates = pd.date_range('2025-01-01 08:00:00', periods=62, freq='12h').strftime('%d/%m') #starting on 01.01.2025, 60 days of observation twice a day
input_data = pd.DataFrame(np.random.randint(0, high=11, size=(62, 5)), index=input_dates, columns=['Mood', 'Activity', 'Energy', 'Symptoms', "Mind clearness"])
dates = pd.date_range('2025-01-01', periods=31, freq='D').strftime('%d/%m')

entries = np.array(range(2, 64))/2
mood = np.array(input_data['Mood'])
activity = np.array(input_data['Activity'])
energy = np.array(input_data['Energy'])
symptoms = np.array(input_data['Symptoms'])
mind_clearness = np.array(input_data['Mind clearness'])

entries_int = np.array(entries, dtype=int)
day = len(np.unique(entries_int))  # day, not the entries amount

# calculation of interim values of MAESC parameters
interim = []
maesc = np.array([mood, activity, energy, symptoms, mind_clearness])
for i in maesc:
        interim.append(np.where(entries_int==entries, i, 11))

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
condition_values = np.array([condition(mood[i], symptoms[i], activity[i]) for i in range(len(entries))])

condition_daily_interim = np.where(entries_int==entries, condition_values, 7)
condition_morning_interim = np.setdiff1d(np.where(entries_int==entries, condition_values, 7), 7, assume_unique=True)
condition_evening = np.setdiff1d(np.where(entries_int!=entries, condition_values, 7), 7, assume_unique=True)

# mean condition and mean MAESC parameters daily calculation
for i in range(len(entries)):
    if condition_daily_interim[i] == 7:
        condition_daily_interim[i - 1] = condition((mood[i]+mood[i-1])/2, (symptoms[i]+symptoms[i-1])/2, (activity[i]+activity[i-1])/2)
        interim[0][i - 1] = (mood[i] + mood[i - 1]) / 2
        interim[1][i - 1] = (activity[i] + activity[i - 1]) / 2
        interim[2][i - 1] = (energy[i] + energy[i - 1]) / 2
        interim[3][i - 1] = (symptoms[i] + symptoms[i - 1]) / 2
        interim[4][i - 1] = (mind_clearness[i] + mind_clearness[i - 1]) / 2
condition_daily = np.setdiff1d(condition_daily_interim, 7, assume_unique=True)
# MAESC daily value
mood_daily = np.setdiff1d(interim[0], 11, assume_unique=True)
activity_daily = np.setdiff1d(interim[1], 11, assume_unique=True)
energy_daily = np.setdiff1d(interim[2], 11, assume_unique=True)
symptoms_daily = np.setdiff1d(interim[3], 11, assume_unique=True)
mind_clearness_daily = np.setdiff1d(interim[4], 11, assume_unique=True)

condition_morning = np.copy(condition_morning_interim)
i = 0
j = 0
k = 0
deleted_items_changes = []
while i < len(entries)-1:
    if entries[i+1] != entries[i]+0.5 and (entries[i] % 2 == 0.0 or entries[i] % 2 == 1.0):
        condition_morning = np.delete(condition_morning, j)
        deleted_items_changes.append(k)
        k += 1
    elif entries[i] % 2 == 0.5 or entries[i] % 2 == 1.5:
        j += 1
    else:
        k += 1
    i += 1
if entries[-1] % 2 == 0.0 or entries[-1] % 2 == 1.0:
    condition_morning = np.delete(condition_morning, -1)

changes = [0 for k in range(15)]
change_daily = [0 for l in range(len(condition_morning))]
for j in range(len(condition_morning)):
    for i in range(-7, 8):
        if int(condition_morning[j]+3)+i == int(condition_evening[j]+3):
            changes[i+7] += 1
            change_daily[j] = i
change_daily = np.array(change_daily)

changes_average = np.mean(change_daily)  # mean value for condition changes
changes_deviation = np.std(change_daily)

entries_change = np.delete(days_list, deleted_items_changes)  # days when changes are possible to calculate
if entries[-1]%2==0.0 or entries[-1]%2==1.0:
    entries_change = np.delete(entries_change, -1)


cr_values = [[i, i+1] for i in range(-3, 5)]
daily_parameters = daily(8, condition_daily, cr_values)


# Activity-energy function into list of values
counter_acten = []
acten_data = []
for i in range(0, day):
    acten_data.append(acten(activity_daily[i], energy_daily[i]))


mean_MAESC_week = []
deviation_MAESC_week = []
mean_condition_week = []
deviation_condition_week = []
mean_condition_changes_week = []
deviation_condition_changes_week = []

values = [[mean_MAESC_week, maesc, entries_int, 'mean'], [deviation_MAESC_week, maesc, entries_int, 'std'], [mean_condition_week, condition_values, entries_int, 'average'], [deviation_condition_week, condition_values, entries_int, 'std'], [mean_condition_changes_week, change_daily, entries_change, 'mean'], [deviation_condition_changes_week, change_daily, entries_change, 'std'], [counter_acten, acten_data, np.unique(entries_int), 'count_nonzero']]

condition_average = float(np.average(condition_values, weights=weghts_for_condition))

weekly_values(values)

list_param = [[] for week in range((entries_int[-1] - 1) // 7)]
# Alert calculation => week evaluation
red_alert = [0 for j in range((entries_int[-1]-1)//7)]
yellow_alert = [0 for j in range((entries_int[-1]-1)//7)]
list_values = [[5.35, 6], [5.25, 5.8], [5, 5.4], [5.8, 5.1], [5.8, 6.25], [1.5, 1.2],
               [2, 1.69], [1.5, 1.7], [0.95, 0.8], [1, 0.88], [3, 2], [0.5, 0.4]]
for week in range((entries_int[-1] - 1) // 7):
    for k in range(len(values)):
        if k == 0:
            for j in range(5):
                list_param[week].append(values[k][0][week][j])
        elif k == 1:
            list_param[week].append(values[k][0][week][0])
            list_param[week].append(values[k][0][week][3])
        else:
            list_param[week].append(values[k][0][week])
    for i in range(12):
        red_alert[week] += alert(list_param[week][i], list_values[i])[0]
        yellow_alert[week] += alert(list_param[week][i], list_values[i])[1]

red_alert = np.array(red_alert)  # red zone weekly
yellow_alert = np.array(yellow_alert)  # yellow zone weekly
green_zone = np.array(12-(red_alert+yellow_alert))  # green zone weekly

# Plots
plt.figure(figsize=(52, 43), dpi=100)
plt.subplots_adjust(hspace=0.5, left=0.03, right=0.97, bottom=0.04, top=0.96)

nums = [1, 2, 3, 4, 5, 9, 8, 7, 10]
x_axis = [entries, entries, entries, entries, entries, entries_change, entries, [[i for i in range(1, (entries_int[-1]-1)//7+1)] for j in range(5)], [[i for i in range(1, (entries_int[-1]-1)//7+1)] for j in range(6)]]
y_axis = [mood, activity, energy, symptoms, mind_clearness, change_daily, condition_values, [[mean_MAESC_week[j][k] for j in range((entries_int[-1]-1)//7)] for k in range(5)], [deviation_condition_week, deviation_condition_changes_week, [deviation_MAESC_week[j][0] for j in range((entries_int[-1]-1)//7)], [deviation_MAESC_week[j][3] for j in range((entries_int[-1]-1)//7)], mean_condition_week, mean_condition_changes_week]]
x_lim_ticks = [[max(entries_int), round(entries_int[-1]/15)], [max(entries_int), round(entries_int[-1]/15)], [max(entries_int), round(entries_int[-1]/15)], [max(entries_int), round(entries_int[-1]/15)], [max(entries_int), round(entries_int[-1]/15)], [max(days_list), round(len(days_list)/15)], [max(entries_int), round(entries_int[-1]/15)], [(entries_int[-1]-1)//7-1, 1], [(entries_int[-1]-1)//7-1, 1]]
y_lim_ticks = [[-0.5, 10.5, 0, 10], [-0.5, 10.5, 0, 10], [-0.5, 10.5, 0, 10], [-0.5, 10.5, 0, 10], [-0.5, 10.5, 0, 10], [-7.5, 7.5, -7, 7], [-3.5, 5.5, -2, 5], [int(np.min(mean_MAESC_week))-1, int(np.max(mean_MAESC_week))+1, int(np.min(mean_MAESC_week))-1, int(np.max(mean_MAESC_week))+1], [int(np.min(mean_condition_changes_week))-1, int(np.max(deviation_MAESC_week[0]))+1, int(np.min(mean_condition_changes_week))-1, int(np.max(deviation_MAESC_week[0]))+1]]
colours = ['#3CB371', '#FFA500', '#FF1493', '#7CFC00', '#FFD700', '#800000', '#800080', ['#3CB371', '#FFA500', '#FF1493', '#7CFC00', '#800080'], ['#00008B', '#800000', '#3CB371', '#7CFC00', '#00afff', 'r']]
cr_values = [6, 5.8, 5.4, 5.8, 5.8, 3.5, 1.5, 1, 1]
av_values = [average[0], average[1], average[2], average[3], average[4], changes_average, condition_average, 500, 500, 500]
titles = ['Mood', 'Activity', 'Energy', 'Symptoms', 'Mind clearness', 'Condition changes', 'Condition', 'Mean weakly MAESC', 'Deviation and mean weekly']
sc = [False, False, False, False, False, True, True, False, False]
maps = [mcolors.ListedColormap(['#0000CD', '#1E90FF', '#00FFFF', 'g', '#FFA500', '#FF4500', '#B22222']), 'nipy_spectral_r']
norms = [mcolors.Normalize(vmin=-7, vmax=7), mcolors.Normalize(vmin=-3.5, vmax=10.7)]
labelss = [dates[::7][i]+'-'+dates[6::7][i] for i in range((entries_int[-1]-1)//7)]
labels_x = [dates[::round(len(days_list)/15)], dates[::round(len(days_list)/15)], dates[::round(len(days_list)/15)], dates[::round(len(days_list)/15)], dates[::round(len(days_list)/15)], dates[::round(len(days_list)/15)], dates[::round(len(days_list)/15)], labelss, labelss]
labels_y = [None, None, None, None, None, None, ['Awful', 'Very bad', 'Bad', 'Danger', 'OK', 'Good', 'Very good', 'Excellent'], None, None]
line_width = [4, 4, 4, 4, 4, 4, 4, [2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]]
labels_plot = [None, None, None, None, None, None, None, ['Mood', 'Activity', 'Energy', 'Symptoms', 'Mind clearness'], ['Deviation Condition', 'Deviation Condition changes', 'Deviation Mood', 'Deviation Symptoms', 'Mean Condition', 'Mean Condition changes']]
markers = [None, None, None, None, None, None, None, ['.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.']]
markers_size = [None, None, None, None, None, None, None, [18, 18, 18, 18, 18], [18, 18, 18, 18, 18, 18]]
amounts = [1, 1, 1, 1, 1, 1, 1, 5, 6]
xlims = [True, True, True, True, True, True, True, False, False]

j = 0
for i in range(len(nums)):
    plotting(nums[i], x_axis[i], y_axis[i], y_lim_ticks[i][0], y_lim_ticks[i][1], y_lim_ticks[i][2], y_lim_ticks[i][3], labels_x[i], labels_y[i], x_lim_ticks[i][0], x_lim_ticks[i][1], colours[i], cr_values[i], av_values[i], titles[i], line_width[i], labels_plot[i], markers[i], markers_size[i], amounts[i], xlims[i])
    if nums[i] == 9:
        plt.axhline(-3.5, c='r', ls='dashed', label='critical value')
    if sc[i]:
        scatt(x_axis[i], y_axis[i], maps[j], norms[j])
        j += 1


data_pie1 = pd.DataFrame([[np.count_nonzero(daily_parameters[i])/day for i in range(8)], ['#770000', '#C60000', 'r', '#FF8C00', '#FFD700', '#87FF00', '#00D700', '#008700'], ['Awful', 'Very bad', 'Bad', 'Danger', 'OK', 'Good', 'Very good', 'Excellent']], index=['plot', 'colors', 'labels'])
data_pie2 = pd.DataFrame([changes, ['#af0087', '#8700af', '#5f00af', '#0000d7', '#0087ff', '#00d7ff', '#00ffaf', 'g', '#d7ff00', '#ffaf00', '#ff5700', '#ff0000', '#c70000', '#870000', '#af005f'], [str(i) for i in range(-7, 8)]], index=['plot', 'colors', 'labels'])
data_pie1_filtered = data_pie1.T[data_pie1.T['plot'] > 0]
data_pie2_filtered = data_pie2.T[data_pie2.T['plot'] > 0]

nums = [11, 12]
colours = [data_pie1_filtered['colors'], data_pie2_filtered['colors']]
plots = [data_pie1_filtered['plot'], data_pie2_filtered['plot']]
labs = [data_pie1_filtered['labels'], data_pie2_filtered['labels']]
tits = ['Condition percentage', 'Condition changes']

for i in range(2):
    plotting_pie(nums[i], labs[i], colours[i], plots[i], tits[i])

plots = [red_alert, yellow_alert, green_zone]
botts = [None, red_alert, red_alert+yellow_alert]
cols = ['r', '#FF8C00', 'g']

plotting_bar(plots, botts, cols)


plt.savefig('image.png')
plt.show()
