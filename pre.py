import numpy
import pandas
import re
from collections import Counter

def one_hot_vector_from_category(input):
    assert(input.dtype == 'category')
    unique_categories = input.cat.categories
    ret = numpy.empty((len(input), len(unique_categories)))
    for index, category in enumerate(unique_categories):
        ret[:, index] = (input == category).astype(float)
    return(ret)

begin = pandas.Timestamp('01/01/2003 00:00:00')
end = pandas.Timestamp('05/13/2015 23:59:00')

rng = pandas.date_range(begin, end, freq = 'min')
decomposed_dates = {
    'hour': rng.hour,
    'day': rng.day,
    'month': rng.month,
    'year': rng.year
}
date_elements = pandas.DataFrame(decomposed_dates, index = rng)

def load_data(filename):
    ret = pandas.read_csv(filename)
    ret['Dates'] = pandas.to_datetime(ret['Dates'])
    return(ret)

def parse_date(raw_data):
    return((raw_data['Dates'] - begin) / (end - begin))

def parse_date_elements(raw_data):
    ret = raw_data\
            .merge(date_elements, left_on = 'Dates', right_index = True)\
            .loc[:, decomposed_dates.keys()]

    return(ret)

def parse_category(raw_data):
    used = [
        False ,True, False, False, True, False, False,  True, False, False,
        False, False, True, True, False, False, True, False, False,  True,
        True,  True, False, True, False, True, False, True,  True, False,
        True, False, True, False,  True, True,  True,  True,  True
    ]
    others_index = 20
    raw_cats = pandas.Series(raw_data['Category'], dtype="category").cat.codes
    for index, raw_cat in enumerate(raw_cats):
        if not used[raw_cat]:
            raw_cats[index] = others_index

    return(pandas.Series(raw_data['Category'], dtype="category").cat.codes)

def parse_day_of_the_week(raw_data):
    return(one_hot_vector_from_category(
        pandas.Series(raw_data['DayOfWeek'], dtype="category")))

def parse_address(raw_data):
    num_common_address = 64 # restrict address size to fit into memory
    address_parser = lambda x: re.sub('[0-9]+ Block of ', '', x).split(' / ')
    address = raw_data['Address'].map(address_parser)
    streets = numpy.array([street for streets in address for street in streets])
    # truncate less common streets to fit into memory
    streets = [street for street, _ in
               Counter(streets).most_common(num_common_address)]
    street_indices = {street: index for index, street in enumerate(streets)}
    street_flags = numpy.zeros((raw_data.shape[0], len(streets)))
    for row_index, address_row in enumerate(address):
        for street in address_row:
            if street in streets:
                column_index = street_indices[street]
                street_flags[row_index][column_index] = 1.
    return(street_flags)

def parse_coordinates(raw_data):
    return(raw_data['X'], raw_data['Y'])

def parse_data(raw_data):
    date = parse_date(raw_data)
    date_elements = parse_date_elements(raw_data)
    x, y = parse_coordinates(raw_data)
    dow = parse_day_of_the_week(raw_data)
    street_flags = parse_address(raw_data)
    result = numpy.hstack((
        numpy.array([date, x, y]).T,
        dow,
        street_flags,
        date_elements))
    return(result)

def partition_flags(data, years):
    return(data.Dates.map(lambda date: date.year in years))

year_partitions = [range(2003, 2016)]

for year_index, years in enumerate(year_partitions):
    raw_train = load_data('train.csv')
    category = parse_category(raw_train)
    valid_flags = partition_flags(raw_train, years)
    # split to yearly data
    raw_train = raw_train[valid_flags]
    category = category[valid_flags]
    numpy.save(
        "train_{}".format(year_index),
        numpy.hstack((numpy.array([category]).T, parse_data(raw_train))).astype(numpy.float16))

    # save memory
    del raw_train

    raw_test = load_data('test.csv')
    raw_test = raw_test[partition_flags(raw_test, years)]
    numpy.save("test_{}".format(year_index), parse_data(raw_test).astype(numpy.float16))
