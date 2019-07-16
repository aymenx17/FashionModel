import csv
import json

'''
This module converts the csv file to a json format. Hence, we save a dictionary where keys are ids and values are metadata per data
example. This afterward during trainig allows us to arbitrary access metadata of one example by using the its id, and therefore
makes easier loading batch of data from the dataset.

'''


# I gave names only to those attributes necessary for a first version of training. We use year to split between test and train.
filenames = ('id', '1', '2', '3', 'categ', '5', '6','year','8','9')

# open and read csv rows as dictionaries
with open('styles.csv') as csvfile:
    reader = csv.DictReader(csvfile, filenames)
    out = json.dumps({ row['id']:row for row in reader})

# write json
with open('styles.json', 'w+') as f:
    f.write(out)

# Let's check if the file works
with open('styles.json', 'r') as f:
    d = json.load(f)
    print(d['7311'])
    print(d['44065'])
