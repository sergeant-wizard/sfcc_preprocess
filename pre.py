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

def parse_date(raw_data):
    return((pandas.to_datetime(raw_data['Dates']) - begin) / (end - begin))

def parse_category(raw_data):
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
    x, y = parse_coordinates(raw_data)
    dow = parse_day_of_the_week(raw_data)
    street_flags = parse_address(raw_data)
    result = numpy.hstack((numpy.array([date, x, y]).T, dow, street_flags))
    return(result)

def split_yearly(data, year):
    return(data.Dates.map(lambda date: int(date[0:4]) == year))

valid_years = range(2003, 2016)

for year in valid_years:
    raw_train = pandas.read_csv('train.csv')
    category = parse_category(raw_train)
    valid_flags = split_yearly(raw_train, year)
    # split to yearly data
    raw_train = raw_train[valid_flags]
    category = category[valid_flags]
    numpy.save(
        "train_{}".format(year),
        numpy.hstack((numpy.array([category]).T, parse_data(raw_train))))

    # save memory
    del raw_train

    raw_test = pandas.read_csv('test.csv')
    raw_test = raw_test[split_yearly(raw_test, year)]
    numpy.save("test_{}".format(year), parse_data(raw_test))
