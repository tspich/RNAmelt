import pandas as pd
import re
import json

#id1+id2_oc_sc

def store_ds(RFU, m, e, data):
    #k = m.group(1)

    #k = e.replace('_', '')
    k = e.replace('.', '')

    oligo_c  = float(m.group(2))
    salt_c   = float(m.group(3))
    
    dataset = {"name": k,
               "oligo_c" : oligo_c,
               "salt_c" : salt_c,
               "data" : list(data[e])
               }

    RFU.append(dataset)

def store_unknown(RFU, e, data):
    k = e.replace('.', '')

    dataset = {"name": k,
               "oligo_c" : 0.5, #None,
               "salt_c" :  150, #None,
               "data" : list(data[e])
               }

    RFU.append(dataset)


# NOTE:
# Suppose that all duplexes of a file are done at the same temperatures.

def read_file_data(filename, sheet=None):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'xlsx':
        xlsx = pd.ExcelFile(filename)
        if sheet:
            df = pd.read_excel(xlsx, sheet_name=sheet)
        else:
            df = pd.read_excel(xlsx)
    elif ext == 'csv':
        df = pd.read_csv(filename)

    RFU = []

    #TODO should change id1+id2 to seq1+seq2?
    #average_pat_mu = re.compile(r"^(\S*)\_(\d*\.\d+|\d+)\_(\d+)")
    average_pat_mu = re.compile(r"^(\S)*\_(\d*\.\d+|\d+)\_(\d+)_\d+")

    T = df['Temperature']
    header = list(df.columns)

    for i, e in enumerate(header[1:], start = 1):
        m = average_pat_mu.match(e)
        #print(m)
        if m:
            store_ds(RFU, m, e, df)
        else:
            store_unknown(RFU, e, df)

    return T, RFU

#TODO eventually add the possibility for more than one dataset for one T set.
def read_data(temp, dataset):
    RFU = []
    T = [float(i) for i in temp.split(',')]
    for data in dataset:
        RFU.append({"name"    : data['name'],
                    "oligo_c" : data['oligo_c'],
                    "salt_c"  : data['salt_c'],
                    "data"    : [float(i) for i in data['data'].split(',')]
                    })

    return T, RFU
