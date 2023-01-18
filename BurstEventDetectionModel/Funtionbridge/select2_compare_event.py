# _*__coding:utf-8 _*__
# @Time :2022/10/5 0005 16:25
# @Author :bay
# @File select2_compare_event.py
# @Software : PyCharm
import os

def read_events():
    event_files = os.listdir('../results/event_values')
    event_files.sort(key=lambda x: float(x[: x.find("_")]))
    print(event_files)
    event_datas = []
    len_events = []
    for event_file in event_files:
        event_path = '../results/event_values/' + event_file
        print(event_path)
        file_data = open(event_path, encoding='utf-8').readlines()
        events_data = [i.strip() for i in file_data]
        len_event = len(events_data)
        event_datas.append(events_data)
        len_events.append(len_event)

    return event_datas, len_events


def compute_bursty_words(event_datas):
    events = event_datas[-1]
    flags = []

    for j_data in event_datas[:-1]:
        flag = 0
        for i in events:
            if i in j_data:
                flag += 1
        flags.append(flag)
    return flags


def zhanbi(len_events, flags):
    len = len_events[-1]
    print(len)
    results = []
    for flag in flags:
        result = round(flag/len, 5)
        results.append(result)
    return results


if __name__ == '__main__':
    # read_events()
    event_datas, len_events = read_events()
    print(event_datas)
    print(len_events[:-1])
    flags = compute_bursty_words(event_datas)
    print(len_events, flags)
    results = zhanbi(len_events, flags)
    print(results)


