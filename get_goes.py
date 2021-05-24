from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from sunpy.time import parse_time, TimeRange

def get_goes_event_list(timerange, goes_class_filter=None):
    """
    Retrieve list of flares detected by GOES within a given time range.

    Parameters
    ----------
    timerange : `sunpy.time.TimeRange`
        The time range to download the event list for.
    goes_class_filter: `str`, optional
        A string specifying a minimum GOES class for inclusion in the list,
        e.g., "M1", "X2".

    Returns
    -------
    `list`:
        A list of all the flares found for the given time range.
    """
    # Importing hek here to avoid calling code that relies on optional dependencies.
    from sunpy.net import hek

    # use HEK module to search for GOES events
    client = hek.HEKClient()
    event_type = 'FL'
    tstart = timerange.start
    tend = timerange.end

    # query the HEK for a list of events detected by the GOES instrument
    # between tstart and tend (using a GOES-class filter)
    if goes_class_filter:
        result = client.search(hek.attrs.Time(tstart, tend),
                               hek.attrs.EventType(event_type),
                               hek.attrs.FL.GOESCls > goes_class_filter,
                               hek.attrs.OBS.Observatory == 'GOES')
    else:
        result = client.search(hek.attrs.Time(tstart, tend),
                               hek.attrs.EventType(event_type),
                               hek.attrs.OBS.Observatory == 'GOES')

    # want to condense the results of the query into a more manageable
    # dictionary
    # keep event data, start time, peak time, end time, GOES-class,
    # location, active region source (as per GOES list standard)
    # make this into a list of dictionaries
    goes_event_list = []

    for r in result:
        goes_event = {
            'event_date': parse_time(r['event_starttime']).strftime(
                '%Y-%m-%d'),
            'start_time': parse_time(r['event_starttime']),
            'peak_time': parse_time(r['event_peaktime']),
            'end_time': parse_time(r['event_endtime']),
            'goes_class': str(r['fl_goescls']),
            'goes_location': (r['event_coord1'], r['event_coord2']),
            'noaa_active_region': r['ar_noaanum']
        }
        goes_event_list.append(goes_event)

    return goes_event_list


years = [2011, 2012, 2013, 2014, 2015, 2017]
for year in tqdm(years):
    tstart = datetime(year=year, month=1, day=1)
    tend = datetime(year=year+1, month=1, day=1)
    timerange = TimeRange(tstart, tend)
    event_list = get_goes_event_list(timerange)
    event_df = pd.DataFrame(event_list)
    event_df.to_csv(f'GOES_event_list_{year}.csv')
