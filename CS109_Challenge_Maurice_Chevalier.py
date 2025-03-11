import random
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import holidays
import calendar
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

TSA_VOLUME_2024 = 904068577
GROWTH_RATE_CURRENT_YEAR = .023
GROWTH_RATE_NEXT_YEAR = .021
NUM_SAMPLES = 10000

def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center;'>TSA PASSENGER TRAFFIC FORECASTOR</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:20px;'>This app predicts the number of passengers on a given day up to 360 days in advance! It also shows how likely"
    "you are to see a cheaper ticket than the current price.</p>", unsafe_allow_html=True)
    
    tsa_df = pd.read_csv("tsa.csv")
    season_df = pd.read_csv("seasonal.csv")
    prices_df = pd.read_csv("prices.csv")
    # Display centered label
    #st.markdown("<h3 style='text-align: center;'>What date are you expecting to travel?</h3>", unsafe_allow_html=True)

    # Set default date as today and restrict past dates
    today = date.today()
    future_date = today + timedelta(days=360)
    select_date, select_trip, ticket_price = st.columns(3)


    season_data = get_season_data(season_df)
    tsa_data = transform_tsa_data(tsa_df)
    holiday_map = get_holiday_factor(tsa_data)
    holiday_factors = standardize_holiday(holiday_map)
    bootstrap_input = generate_factors(season_data, tsa_data)

    with select_date:
        select_date = st.date_input("Select your travel date: 📅", min_value=today, max_value=future_date)

    with select_trip:
        options = ["Transatlantic", "Transpacific", "Latin America", "Caribbean", "Mexico", "Transcontinental", "Hawaii", "Intra West Coast", "Florida"]
        selected_option = st.selectbox("Choose an option:", options)
        #st.write(f"You selected: {selected_option}")
    


    check_date = select_date
    # Display selected date
    if check_date:
        formatted_date = check_date.strftime("%B %d, %Y")  # Example: "March 9, 2025"
        st.markdown(f"""
            <h3 style="text-align: center; color: green;">
                ✈️ Your Travel Date: {formatted_date} ✈️
            </h3>
        """, unsafe_allow_html=True)
    
    
    check_date = check_date.strftime("%m/%d/%Y")
    price_month = date_char(check_date)[3]
    day_factors = bootstrap_day_factor(check_date, bootstrap_input)
    month_factors = bootstrap_month_factor(check_date, bootstrap_input)
    pass_forecast = "{:,}".format(predict_volume(check_date, month_factors, day_factors, holiday_factors))
    price_map = get_prices(prices_df)
    lst_prices = generate_price_lst(price_month, selected_option, price_map)
    sampled_prices = bootstrap_price(lst_prices)
    sample_mean_of_price_list = int(np.mean(sampled_prices))
    sample_std_of_price_list = (np.std(sampled_prices, ddof=0))
    
    with ticket_price:
        current_price = st.number_input("Ticket Price", min_value=0, value=sample_mean_of_price_list)
    prob_cheaper_tix = stats.norm.cdf(current_price, sample_mean_of_price_list, sample_std_of_price_list)
    st.markdown(f"<p style='text-align: center; font-size:50px; color:blue'>An estimated {pass_forecast} passengers will travel through TSA on {check_date}.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size:50px; color:red'>There is a {round(prob_cheaper_tix * 100, 2)}% probability of a ticket less than ${current_price} based on historical pricing.</p>", unsafe_allow_html=True)
    plot_figures(day_factors, month_factors, current_price, sampled_prices)
    


def get_holiday_factor(tsa):
    holiday_dict = {}
    for date, chars in tsa.items():
        year = int(date[-4:])
        celebrate = chars[2]
        if celebrate != "No Holiday" and year != 2020 and year != 2021 and year != 2022:
            if celebrate not in holiday_dict:
                holiday_dict[celebrate] = [0, 0]
            holiday_dict[celebrate][0] += int(chars[3].replace(",", ""))
            holiday_dict[celebrate][1] += 1
    return holiday_dict

def standardize_holiday(holiday_map):
    std_holiday_map = {}
    finalized_holiday_map = {}
    for holiday, traffic in holiday_map.items():
        if holiday[-11:] == " (observed)":
            holiday = holiday[:-11]
        if holiday not in std_holiday_map:
            std_holiday_map[holiday] = traffic
            continue
        std_holiday_map[holiday][0] += traffic[0]
        std_holiday_map[holiday][1] += traffic[1]
    for holiday, traffic in std_holiday_map.items():
        finalized_holiday_map[holiday] = traffic[0] / traffic[1]
    return finalized_holiday_map


def predict_volume(date, mon_factors_lst, day_factors_lst, holi_days):
    growth_rate = get_rate(date)
    month_factor = random.choices(mon_factors_lst, k=1)[0]
    day_factor = random.choices(day_factors_lst, k=1)[0]
    day_divisor = get_day_divisor(date)
    traffic = TSA_VOLUME_2024 * growth_rate * month_factor * day_factor * (1 / day_divisor)
    holiday_name =  is_holiday(date)
    if holiday_name in holi_days:
        holiday_factor = holi_days[holiday_name] / traffic
    holiday_factor = 1
    return round(traffic * holiday_factor)


def get_rate(date):
    year = int(date_char(date)[2])
    current_year =  datetime.now().year
    if year == current_year:
        return 1 + GROWTH_RATE_CURRENT_YEAR
    return 1 + GROWTH_RATE_NEXT_YEAR
    

def plot_figures(day, month, price, sample_prices):
    
    fig_1 = plt.figure()
    plot_day_factor(day)
    fig_2 = plt.figure()
    plot_price_prob(price, sample_prices)
    fig_3 = plt.figure()
    plot_month_factor(month)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.pyplot(fig_1)
    with col2:
        st.pyplot(fig_2)
    with col3:
        st.pyplot(fig_3)

def plot_day_factor(factors):
    plt.hist(factors, bins=3, color='purple', edgecolor='black')
    plt.xlabel("Day Factors")
    plt.ylabel("Frequency")
    plt.title("Day Factors Histogram")


def plot_month_factor(factors):
    plt.hist(factors, bins=100, color='orange', edgecolor='black')
    plt.xlabel("Month Factors")
    plt.ylabel("Frequency")
    plt.title("Month Factors Histogram")


def plot_price_prob(current_price, sampled_prices):
    plt.hist(sampled_prices, bins=100, color='green', edgecolor='black')
    plt.xlabel("Prices")
    plt.ylabel("Frequency")
    plt.title("Price Histogram")
    plt.axvline(x=current_price, color='r', linestyle='--', linewidth=2, label="Current Price")


def bootstrap_day_factor(date, boot_input_dict):
    date_chars = date_char(date)
    month_name, day_name = date_chars[3], date_chars[4]
    sample = 0
    sample_day_factor = []
    n_day_samples = 3
    day_samples = boot_input_dict[month_name]["DAY_FACTOR"][day_name]
    while sample < NUM_SAMPLES:
        mean_day_factor = np.mean(random.choices(day_samples, k=n_day_samples))
        sample_day_factor.append(round(float(mean_day_factor), 5))
        sample += 1
    return sample_day_factor


def bootstrap_month_factor(date, boot_input_dict):
    date_chars = date_char(date)
    month_name = date_chars[3]
    sample = 0
    sample_month_factor = []
    n_mon_samples = 19
    mon_samples = boot_input_dict[month_name]["MONTH_FACTOR"]
    while sample < NUM_SAMPLES:
        mean_month_factor = np.mean(random.choices(mon_samples, k=n_mon_samples))
        sample_month_factor.append(round(float(mean_month_factor), 5))
        sample += 1
    return sample_month_factor


# this function provides the reference inputs for the bootstrap function to work
# takes in monthly traffic levels of passengers since 2003 and daily traffic levels
# that is put out by tsa since 2019. 
def generate_factors(season_data, tsa_data): 
    prediction_factor = {}
    for month, years in season_data.items():
        for year in years:
            month_factor = season_data[month][year][2]
            if month not in prediction_factor:
                prediction_factor[month] = {"MONTH_FACTOR": [month_factor]}
            else:
                prediction_factor[month]["MONTH_FACTOR"].append(month_factor)
            if is_year_in_tsa(year, tsa_data):
                day_factors = get_day_factor(month, year, tsa_data)
                for day in day_factors:
                    if "DAY_FACTOR" not in prediction_factor[month]:
                        prediction_factor[month]["DAY_FACTOR"] = {}
                    if day not in prediction_factor[month]["DAY_FACTOR"]:
                        prediction_factor[month]["DAY_FACTOR"][day] = [day_factors[day]]
                    else:
                        prediction_factor[month]["DAY_FACTOR"][day].append(day_factors[day])
    return prediction_factor


def is_year_in_tsa(year, tsa):
    for date in tsa:
        check_year = int(date[-4:])
        if year == check_year: return 1
    return 0

def date_char(date):
    first_slash = date.find("/")
    second_slash = date.find("/", first_slash + 1)
    day = date[first_slash + 1:second_slash]
    year = date[-4:]
    month = date[:2]
    month_name = get_month_str(month)
    day_name = calendar.day_name[calendar.weekday(int(year), (int(month)), int(day))]
    return [month, day, year, month_name, day_name]



def get_day_factor(check_month, check_year, tsa):
    count_days_in_month = {"Monday": [0, 0], "Tuesday": [0, 0], "Wednesday": [0, 0], "Thursday": [0, 0], "Friday": [0, 0], "Saturday": [0, 0], "Sunday": [0, 0]}
    monthly_tsa_passengers = 0
    day_factor = {}
    for date, date_char in tsa.items():
        year = int(date[-4:])
        month = date_char[0]
        day = date_char[1]
        traffic =  int(date_char[3].replace(",", ""))
        if month == check_month and year == check_year:
            monthly_tsa_passengers += traffic
            count_days_in_month[day][0] += traffic
            count_days_in_month[day][1] += 1

    if monthly_tsa_passengers != 0:
        avg_total_week = 0
        day_factor_elem = {}
        for day, passengers in count_days_in_month.items():
            day_average = passengers[0] / passengers[1]
            avg_total_week += day_average
            day_factor_elem[day] = day_average
        for day in day_factor_elem:
            day_factor[day] = round(day_factor_elem[day] / avg_total_week, 5)
        return day_factor
    return
        
   
def transform_tsa_data(tsa_df):
    tsa_data = {}
    for data in tsa_df.itertuples():
        date_obj = datetime.strptime(data[1], "%m/%d/%Y")
        date = date_obj.strftime("%m/%d/%Y")
        day = datetime.strptime(date, "%m/%d/%Y").strftime("%A")
        month = date_obj.strftime("%B")
        traffic = data[2]
        tsa_data[date] = [month, day, is_holiday(date), traffic]
    return tsa_data 


def get_season_data(seasonal_df):
    season_data = {}
    for data in seasonal_df.itertuples():
        month = data[2]
        current_year = int(data[1])
        domes_count = int(data[3])
        intl_count = int(data[4])
        total_count = int(data[5])
        if month != "TOTAL" and current_year != 2020 and current_year != 2021 and current_year != 2022:
            month = get_month(int(month))
            if month not in season_data:
                season_data[month] = {current_year: [domes_count, intl_count, domes_count + intl_count]}
            else:
                season_data[month][current_year] = [domes_count, intl_count, domes_count + intl_count]
        else:
            for each_month, passengers in season_data.items():
                for year in passengers:
                    if current_year == year:
                        domestic_pass = season_data[each_month][year][0]
                        intl_pass = season_data[each_month][year][1]
                        total_pass = season_data[each_month][year][2]
                        season_data[each_month][year][0] = round(domestic_pass / domes_count, 5)
                        season_data[each_month][year][1] = round(intl_pass / intl_count, 5)
                        season_data[each_month][year][2] = round(total_pass / total_count, 5)
    return season_data


def is_holiday(date):
    year = date[-4:]
    for holiday, name in holidays.US(years=year).items():
        holiday = holiday.strftime("%m/%d/%Y")
        if date == holiday:
            return name
    return "No Holiday"


def get_month(num):
    months = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
    return months[num]


def get_month_str(month_num):
    month_dict = {"01": "January", "02": "February", "03": "March", "04": "April", "05": "May", "06": "June", "07": "July", "08": "August", "09": "September", "10": "October", "11": "November", "12": "December"}
    return next((name for month, name in month_dict.items() if month == month_num), "Invalid Month")

def get_day_divisor(date):
    date_chars = date_char(date)
    month = date_chars[0]
    year = int(date_chars[2])
    thirty = ["April", "June", "September", "November"]
    thirty_one = ["January", "March", "May", "July", "August", "October", "December"]
    if month in thirty:
        return 30 / 7
    if month in thirty_one:
        return 31 / 7
    if calendar.isleap(year):
        return 29 / 7
    return 28 / 7


def generate_price_lst(month, region, price_map):
    return price_map[month][region]


def bootstrap_price(price_lst):
    sample =  0
    sampled_prices = []
    n_prices = len(price_lst)
    while sample < NUM_SAMPLES:
        mean_price = np.mean(random.choices(price_lst, k=n_prices))
        sampled_prices.append(round(float(mean_price), 2))
        sample += 1
    return sampled_prices


def get_prices(prices_df):
    region_price_map = {}
    region_key = {0: "Transatlantic", 1: "Transpacific", 2: "Latin America", 3: "Caribbean", 4: "Mexico", 5: "Transcontinental", 6: "Hawaii", 7: "Intra West Coast", 8: "Florida"} 
    clean_prices = prices_df.dropna()
    for index, region in region_key.items():
        x = 0
        for date, price in clean_prices.loc[index].items():
            if date == "Unnamed: 0":
                continue
            cost = int(price.replace(",", ""))
            first_slash = date.find("/")
            month = int(date[:first_slash])
            month_name = get_month(month)
            if month_name not in region_price_map:
                region_price_map[month_name] = {region: [cost]}
            elif region not in region_price_map[month_name]:
                region_price_map[month_name][region] = [cost] 
            else:
                region_price_map[month_name][region].append(cost)
    return region_price_map
        

if __name__ == '__main__':
    main()