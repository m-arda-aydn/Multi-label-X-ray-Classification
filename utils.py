def return_seconds(seconds):
    days, seconds = divmod(seconds, 86400)  
    hours, seconds = divmod(seconds, 3600)  
    minutes, seconds = divmod(seconds, 60)
    date_dict = {"days": days, "hours": hours, "minutes": minutes, "seconds": seconds}
    new_dict = dict()
    for key, value in date_dict.items():
        if value == 0:
            continue
        new_dict[key] = int(value)
    return new_dict