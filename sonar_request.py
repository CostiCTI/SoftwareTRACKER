import requests
import pprint as pp


def get_sonar_historical_data(project_name):
    '''
        This function make a request to sonar in order to get all historical values
    for software metrics

    Params:
    ------
    project_name: string - name of the project that we want to analyse

    Returns:
    data: list - contains date and value for each metric, for each version of
                 the project
    '''

    # metrics list
    metrics = ['ncloc', 'complexity', 'bugs', 'code_smells', 'comment_lines', 
               'classes', 'files', 'functions', 'violations', 'major_violations',
               'minor_violations', 'vulnerabilities', 'lines']

    comp = project_name
    mlist = ""
    for m in metrics:
        mlist = mlist + m + ","
    mlist = mlist[:-1]

    URL = "http://localhost:9000/api/measures/search_history?component=flexx&metrics=" + mlist
    
    r = requests.get(url = URL)
    
    resp = r.json()
    if ('measures' not in resp):
        return None

    data = []
    rs = resp['measures']
    for el in rs:
        historical_data = el['history']
        historical_data.sort(key=lambda item:item['date'])
        data.append((el['metric'], historical_data))

    return data;


def get_sonar_current_data(project_name):
    '''
        This function make a request to sonar in order to get last version values
    for software metrics

    Params:
    ------
    project_name: string - name of the project that we want to analyse

    Returns:
    data: dict - contains value for each metric
    '''

    # metrics list
    metrics = ['ncloc', 'complexity', 'bugs', 'code_smells', 'comment_lines', 
               'classes', 'files', 'functions', 'violations', 'major_violations',
               'minor_violations', 'vulnerabilities', 'lines']

    comp = project_name
    mlist = ""
    for m in metrics:
        mlist = mlist + m + ","
    mlist = mlist[:-1]

    URL = "http://localhost:9000/api/measures/component?metricKeys=" + mlist + "&component=" + comp
    
    r = requests.get(url = URL)
    
    resp = r.json()
    if ('component' not in resp):
        return None

    metric_list = resp['component']['measures']
    data = {}
    for d in metric_list:
        data[d['metric']] = d['value']

    return data


if __name__ == '__main__':
    data = get_sonar_historical_data('project_name')
    pp.pprint(data)
    data = get_sonar_current_data('project_name')
    pp.pprint(data)