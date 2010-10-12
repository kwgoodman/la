# This code was taken from the matplotlib trunk on 2010-10-11
#
# See la/LICENSE for the matplotlib license
#
# The original matplotlib code:
# Copyright (c) 2002-2009 John D. Hunter; All Rights Reserved
#
# The following modification were made (all modifications are
# covered by the la license):
#
#   - Remove verbose class
#   - Remove caching
#   - Remove year, month, day, d (date number); just return datetime.date
#   - Remove rearrays and asobject parameter; always return lists of tuples
#   - Don't adjust prices or volume, leave that for the la package
#   - parse_historical_yahoo now returns dates and data separately
#   - quotes_historical_yahoo now returns dates and data separately
#   - parse_historical_yahoo no longer reverses data order
#   - Modified doc strings to reflect above changes

"Load Yahoo! Finance data from the web"

import warnings
from urllib2 import urlopen
import datetime


def parse_yahoo_historical(fh):
    """
    Parse the historical data in file handle fh from yahoo finance.

    Return a list of tuples containing

        date, open, close, high, low, volume, adj close

    where date is a python standard library datetime.date instance.

    """
    lines = fh.readlines()
    data = []
    dates = []
    datefmt = '%Y-%m-%d'
    for line in lines[1:]:
        vals = line.split(',')
        if len(vals)!=7:
            continue      # add warning?
        datestr = vals[0]
        date = datetime.date(*[int(val) for val in datestr.split('-')])
        dates.append(date)
        open, high, low, close = [float(val) for val in vals[1:5]]
        volume = int(vals[5])
        aclose = float(vals[6])
        data.append((open, close, high, low, volume, aclose))
    return data, dates 

def fetch_historical_yahoo(ticker, date1, date2):
    """
    Fetch historical data for ticker between date1 and date2.  date1 and
    date2 are date or datetime instances, or (year, month, day) sequences.

    Ex:
    fh = fetch_historical_yahoo('^GSPC', (2000, 1, 1), (2001, 12, 31))

    a file handle is returned
    """
    ticker = ticker.upper()
    if iterable(date1):
        d1 = (date1[1]-1, date1[2], date1[0])
    else:
        d1 = (date1.month-1, date1.day, date1.year)
    if iterable(date2):
        d2 = (date2[1]-1, date2[2], date2[0])
    else:
        d2 = (date2.month-1, date2.day, date2.year)
    urlFmt = 'http://table.finance.yahoo.com/table.csv?a=%d&b=%d&c=%d&d=%d&e=%d&f=%d&s=%s&y=0&g=d&ignore=.csv'
    url =  urlFmt % (d1[0], d1[1], d1[2],
                     d2[0], d2[1], d2[2], ticker)
    return urlopen(url)

def quotes_historical_yahoo(ticker, date1, date2):
    """
    Get historical data for ticker between date1 and date2.
    
    date1 and date2 are datetime instances or (year, month, day) sequences.

    See :func:`parse_yahoo_historical` for explanation of output formats
    and the *asobject* and *adjusted* kwargs.

    Ex:
    sp = f.quotes_historical_yahoo('^GSPC', d1, d2)

    """
    fh = fetch_historical_yahoo(ticker, date1, date2)
    try:
        data, dates = parse_yahoo_historical(fh)
        if len(data) == 0:
            return None
    except IOError, exc:
        warnings.warn('urlopen() failure\n' + exc.strerror[1])
        return None
    return data, dates

def iterable(obj):
    'return true if *obj* is iterable'
    try: len(obj)
    except: return False
    return True
